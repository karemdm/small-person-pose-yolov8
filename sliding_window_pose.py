import argparse
import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torchvision.ops as ops


def sliding_window(image, window_size, overlap):
    """
    Yield (x1, y1, x2, y2) tiles that cover the whole image.
    Tiles on the right/bottom edges are clipped to the image size.
    """
    step = max(int(window_size * (1 - overlap)), 1)
    h, w, _ = image.shape

    for y in range(0, h, step):
        for x in range(0, w, step):
            x1 = x
            y1 = y
            x2 = min(x + window_size, w)  # do not go past the image width
            y2 = min(y + window_size, h)  # do not go past the image height
            yield x1, y1, x2, y2


def adjust_coordinates(detections, x_offset, y_offset):
    """
    Shift tile boxes (x1, y1, x2, y2) back to frame coordinates.
    """
    if detections is None or detections.size(0) == 0:
        return detections

    detections = detections.clone()
    detections[:, [0, 2]] += x_offset
    detections[:, [1, 3]] += y_offset
    return detections


def adjust_keypoints(keypoints, x_offset, y_offset):
    """
    Shift tile keypoints back to frame coordinates,
    leaving missing keypoints (x or y == 0) untouched.
    """
    if keypoints is None or keypoints.size(0) == 0:
        return keypoints

    keypoints = keypoints.clone()

    keypoints[:, :, 0] = torch.where(keypoints[:, :, 0] != 0, keypoints[:, :, 0] + x_offset, keypoints[:, :, 0])
    keypoints[:, :, 1] = torch.where(keypoints[:, :, 1] != 0, keypoints[:, :, 1] + y_offset, keypoints[:, :, 1])

    return keypoints


def merge_detections(all_boxes, all_keypoints, iou_threshold=0.5):
    """
    Concatenate the detections of all tiles and remove duplicates with NMS.

    Each box row is [x1, y1, x2, y2, (track_id,) conf, cls], so the confidence is
    always the second-to-last column. Returns (None, None) when there is nothing to merge.
    """
    if len(all_boxes) == 0:
        return None, None

    target_device = all_boxes[0].device
    combined_boxes = torch.cat([box.to(target_device) for box in all_boxes], dim=0)
    combined_keypoints = torch.cat([kpt.to(target_device) for kpt in all_keypoints], dim=0)

    keep = ops.nms(combined_boxes[:, :4], combined_boxes[:, -2], iou_threshold)
    return combined_boxes[keep], combined_keypoints[keep]


def collect_tile_detections(result, x_offset, y_offset, all_boxes, all_keypoints):
    """
    Append the detections of one tile (shifted to frame coordinates) as
    [x1, y1, x2, y2, conf, cls] rows. Tiles where no person has at least
    2 visible keypoints are skipped.
    """
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return
    non_zero_keypoints = (result.keypoints.xy != 0).sum(dim=(1, 2))
    if not (non_zero_keypoints >= 2).any():
        return

    boxes = adjust_coordinates(result.boxes.xyxy, x_offset, y_offset)
    keypoints = adjust_keypoints(result.keypoints.xy, x_offset, y_offset)
    confidence = result.boxes.conf
    class_id = result.boxes.cls
    # Track IDs of the per-tile tracker are not kept: they are not meaningful across tiles,
    # and mixing 6- and 7-column rows would break torch.cat / NMS.
    all_boxes.append(torch.cat((boxes, confidence.unsqueeze(1), class_id.unsqueeze(1)), dim=1))
    all_keypoints.append(keypoints)


def render_frame(frame, all_boxes, all_keypoints):
    """Merge the tile detections of one frame and draw them; return the frame unchanged if empty."""
    combined_boxes, combined_keypoints = merge_detections(all_boxes, all_keypoints)
    if combined_boxes is not None and combined_boxes.size(0) > 0 and \
            not (combined_keypoints == 0).all():
        results = Results(
            orig_img=frame,
            boxes=combined_boxes,
            keypoints=combined_keypoints,
            names={0: "person"},
            path=""
        )
        return results.plot()
    return frame


def open_video_io(video_path, output_path):
    """Open the input video and an mp4v writer with the same size/FPS; return (cap, out) or (None, None)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        return None, None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # some containers report 0 FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: cannot open the output video for writing: {output_path}")
        cap.release()
        return None, None
    return cap, out


def process_video(video_path, model_path, window_size, overlap, output_path):
    cap, out = open_video_io(video_path, output_path)
    if cap is None:
        return
    model = YOLO(model_path)

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

    while True:
        ret, frame = cap.read()
        if not ret or frame_number == 100:
            break

        all_boxes = []
        all_keypoints = []

        for x1, y1, x2, y2 in sliding_window(frame, window_size, overlap):
            window = frame[y1:y2, x1:x2]
            results = model.track(window, device=0, verbose=False, conf=0.3, iou=0.8)
            collect_tile_detections(results[0], x1, y1, all_boxes, all_keypoints)

        out.write(render_frame(frame, all_boxes, all_keypoints))
        frame_number += 1
        print(f"Processed frame: {frame_number}", end="\r")

    print(f"Processed frame: {frame_number}")
    cap.release()
    out.release()
    print("Processing complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Sliding-window YOLOv8 pose estimation on a video.")
    parser.add_argument("--source", required=True, help="input video path")
    parser.add_argument("--weights", default="yolov8n-pose.pt", help="YOLOv8 pose weights")
    parser.add_argument("--output", default="processed_video.mp4", help="output video path")
    parser.add_argument("--window-size", type=int, default=640, help="sliding window size (px)")
    parser.add_argument("--overlap", type=float, default=0.2, help="overlap between windows (0-1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args.source, args.weights, args.window_size, args.overlap, args.output)
