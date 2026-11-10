import argparse
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from ultralytics.engine.results import Results

def dynamic_crop_and_recombine(image, detections, model_pose, padding=50):
    """
    Run the pose model on a padded crop around each detection and draw the results on the original frame.

    Args:
        image (np.ndarray): Original frame.
        detections (list): YOLO detections as [x1, y1, x2, y2, conf, cls].
        model_pose: YOLO pose model run on each crop.
        padding (int): Padding (px) added around each detection.

    Returns:
        np.ndarray: Annotated frame (the input frame if no pose was found).
    """
    h, w, _ = image.shape

    # Detections of all crops, in frame coordinates
    combined_boxes = []
    combined_keypoints = []

    for det in detections:
        # Box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, det[:4])

        # Pad the box
        x1_crop = max(x1 - padding, 0)
        y1_crop = max(y1 - padding, 0)
        x2_crop = min(x2 + padding, w)
        y2_crop = min(y2 + padding, h)

        # Crop the frame
        crop = image[y1_crop:y2_crop, x1_crop:x2_crop]

        # Run the pose model on the crop
        crop_results = model_pose.predict(source=crop, conf=0.2, iou=0.5,
                                          verbose=False, device=0, imgsz=128)
        crop_boxes = crop_results[0].boxes.data.cpu().numpy()
        crop_keypoints = crop_results[0].keypoints.data.cpu().numpy()

        if len(crop_boxes) == 0:
            continue  # no person found in this crop

        # Shift the crop detections back to frame coordinates
        for i, crop_det in enumerate(crop_boxes):
            # Boxes with only 4 values get a default confidence and class
            if len(crop_det) == 4:
                cx1, cy1, cx2, cy2 = crop_det
                cconf = 1.0
                ccls = 0
            else:
                cx1, cy1, cx2, cy2, cconf, ccls = crop_det

            cx1 += x1_crop
            cy1 += y1_crop
            cx2 += x1_crop
            cy2 += y1_crop
            combined_boxes.append([cx1, cy1, cx2, cy2, cconf, ccls])

            # Shift the keypoints
            keypoints = crop_keypoints[i]
            keypoints[:, 0] += x1_crop
            keypoints[:, 1] += y1_crop
            combined_keypoints.append(keypoints)

    if len(combined_boxes) == 0:
        return image  # nothing to draw

    combined_boxes_array = np.array(combined_boxes, dtype=np.float32)
    combined_keypoints_array = np.array(combined_keypoints, dtype=np.float32)

    combined_boxes_tensor = torch.tensor(combined_boxes_array, dtype=torch.float32, device='cuda:0')
    combined_keypoints_tensor = torch.tensor(combined_keypoints_array, dtype=torch.float32, device='cuda:0')

    results = Results(
        orig_img=image,
        boxes=combined_boxes_tensor,
        keypoints=combined_keypoints_tensor,
        names={0: 'person'},
        path=""
    )

    annotated_image = results.plot()

    return annotated_image


def process_video(video_path, model, model_pose, output_path):
    """
    Process a video frame by frame: detect people on the full frame, run the pose
    model on a crop around each one and write the annotated video.

    Args:
        video_path (str): Input video path.
        model: YOLO detection model (full frame).
        model_pose: YOLO pose model (crops).
        output_path (str): Output video path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # some containers report 0 FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: cannot open the output video for writing: {output_path}")
        cap.release()
        return

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_number == 800:
            break

        # Person detection on the full frame
        results_full = model.predict(source=frame, conf=0.25, iou=0.5, classes=[0], imgsz=1280,
                                     verbose=False, device=0)

        initial_detections = results_full[0].boxes.data.cpu().numpy()

        if len(initial_detections) > 0:
            # Pose on a crop around each person, drawn back on the frame
            annotated_frame = dynamic_crop_and_recombine(frame, initial_detections, model_pose, padding=50)
        else:
            annotated_frame = results_full[0].plot()

        out.write(annotated_frame)
        frame_number += 1
        print(f"Processed frame: {frame_number}", end="\r", flush=True)

    print()
    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic-crop YOLOv8 pose estimation on a video.")
    parser.add_argument("--source", required=True, help="input video path")
    parser.add_argument("--output", default="./output_video.mp4", help="output video path")
    parser.add_argument("--det-weights", default="yolov8x.pt", help="YOLOv8 detection weights")
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt", help="YOLOv8 pose weights")
    args = parser.parse_args()

    model = YOLO(args.det_weights)
    model_pose = YOLO(args.pose_weights)

    process_video(args.source, model, model_pose, args.output)
