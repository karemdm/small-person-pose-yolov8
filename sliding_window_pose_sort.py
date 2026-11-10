import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torchvision.ops as ops
from sort import Sort

# Shared helpers (tiling, coordinate shift, NMS merge)
from sliding_window_pose import (
    collect_tile_detections,
    merge_detections,
    open_video_io,
    sliding_window,
)


def track_and_render(frame, all_boxes, all_keypoints, tracker, match_iou=0.3):
    """
    Merge the tile detections of one frame, update SORT and draw the detections
    that SORT returned as confirmed tracks, labelled with their track ID.
    """
    combined_boxes, combined_keypoints = merge_detections(all_boxes, all_keypoints)

    if combined_boxes is None or combined_boxes.size(0) == 0:
        tracker.update(np.empty((0, 5)))  # still age the existing tracks
        return frame

    # SORT input: [x1, y1, x2, y2, score]
    dets = torch.cat((combined_boxes[:, :4], combined_boxes[:, 4:5]), dim=1).cpu().numpy()
    track_bbs_ids = tracker.update(dets)  # [x1, y1, x2, y2, track_id]
    if len(track_bbs_ids) == 0:
        return frame

    # SORT returns Kalman-filtered boxes, so match them back to the detections by IoU
    track_boxes = torch.as_tensor(track_bbs_ids[:, :4], dtype=combined_boxes.dtype, device=combined_boxes.device)
    track_ids = torch.as_tensor(track_bbs_ids[:, 4], dtype=combined_boxes.dtype, device=combined_boxes.device)
    best_iou, best_track = ops.box_iou(combined_boxes[:, :4], track_boxes).max(dim=1)
    matched = best_iou >= match_iou
    if not matched.any():
        return frame

    boxes = combined_boxes[matched]
    keypoints = combined_keypoints[matched]
    # Ultralytics box layout with tracking: [x1, y1, x2, y2, track_id, conf, cls]
    boxes = torch.cat((boxes[:, :4], track_ids[best_track[matched]].unsqueeze(1), boxes[:, 4:6]), dim=1)

    if (keypoints == 0).all():
        return frame
    results = Results(
        orig_img=frame,
        boxes=boxes,
        keypoints=keypoints,
        names={0: "person"},
        path=""
    )
    return results.plot()


def process_video_batch(video_path, model_path, window_size, overlap, output_path, batch_size=16):
    """
    Batched sliding-window YOLOv8-pose + SORT tracking (persistent IDs across frames).
    """
    cap, out = open_video_io(video_path, output_path)
    if cap is None:
        return
    model = YOLO(model_path)
    tracker = Sort()

    def process_batch(frames_batch, window_batch, window_positions):
        windows_per_frame = len(window_positions) // len(frames_batch)
        results = model(window_batch, device=0, verbose=False, conf=0.3, iou=0.8)
        for frame_idx, frame in enumerate(frames_batch):
            all_boxes = []
            all_keypoints = []
            start = frame_idx * windows_per_frame
            for idx in range(start, start + windows_per_frame):
                x_offset, y_offset = window_positions[idx]
                collect_tile_detections(results[idx], x_offset, y_offset, all_boxes, all_keypoints)
            out.write(track_and_render(frame, all_boxes, all_keypoints, tracker))

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    frames_batch = []
    window_batch = []  # tiles of all buffered frames
    window_positions = []  # (x1, y1) offset of each tile

    while True:
        ret, frame = cap.read()
        if not ret or frame_number == 100:
            break

        frames_batch.append(frame)
        frame_number += 1

        for x1, y1, x2, y2 in sliding_window(frame, window_size, overlap):
            window_batch.append(frame[y1:y2, x1:x2])
            window_positions.append((x1, y1))

        if len(frames_batch) == batch_size:
            process_batch(frames_batch, window_batch, window_positions)
            print(f"Processed frame: {frame_number}", end="\r")
            frames_batch, window_batch, window_positions = [], [], []

    # Last, partial batch
    if frames_batch:
        process_batch(frames_batch, window_batch, window_positions)
        print(f"Processed frame: {frame_number}", end="\r")

    print("\nProcessing complete.")
    cap.release()
    out.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Batched sliding-window YOLOv8 pose estimation with SORT tracking.")
    parser.add_argument("--source", required=True, help="input video path")
    parser.add_argument("--weights", default="yolov8n-pose.pt", help="YOLOv8 pose weights")
    parser.add_argument("--output", default="processed_video.mp4", help="output video path")
    parser.add_argument("--window-size", type=int, default=640, help="sliding window size (px)")
    parser.add_argument("--overlap", type=float, default=0.2, help="overlap between windows (0-1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video_batch(args.source, args.weights, args.window_size, args.overlap, args.output)
