import argparse
import cv2
from ultralytics import YOLO

# Shared helpers (tiling, coordinate shift, NMS merge, drawing)
from sliding_window_pose import (
    collect_tile_detections,
    open_video_io,
    render_frame,
    sliding_window,
)


def process_batch(model, frames_batch, window_batch, window_positions, out):
    """
    Run the model once on all tiles of the buffered frames, then merge and
    draw the detections frame by frame and write the frames in order.
    """
    windows_per_frame = len(window_positions) // len(frames_batch)
    results = model.track(window_batch, device=0, verbose=False, conf=0.3, iou=0.8)

    for frame_idx, frame in enumerate(frames_batch):
        all_boxes = []
        all_keypoints = []
        start = frame_idx * windows_per_frame
        for idx in range(start, start + windows_per_frame):
            x_offset, y_offset = window_positions[idx]
            collect_tile_detections(results[idx], x_offset, y_offset, all_boxes, all_keypoints)
        out.write(render_frame(frame, all_boxes, all_keypoints))


def process_video_batch(video_path, model_path, window_size, overlap, output_path, batch_size=16):
    """
    Sliding-window YOLOv8-pose where the tiles of `batch_size` frames are sent to the model in one call.
    """
    cap, out = open_video_io(video_path, output_path)
    if cap is None:
        return
    model = YOLO(model_path)

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
            process_batch(model, frames_batch, window_batch, window_positions, out)
            print(f"Processed frame: {frame_number}", end="\r")
            frames_batch, window_batch, window_positions = [], [], []

    # Last, partial batch
    if frames_batch:
        process_batch(model, frames_batch, window_batch, window_positions, out)
        print(f"Processed frame: {frame_number}", end="\r")

    print("\nProcessing complete.")
    cap.release()
    out.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Batched sliding-window YOLOv8 pose estimation on a video.")
    parser.add_argument("--source", required=True, help="input video path")
    parser.add_argument("--weights", default="yolov8n-pose.pt", help="YOLOv8 pose weights")
    parser.add_argument("--output", default="processed_video.mp4", help="output video path")
    parser.add_argument("--window-size", type=int, default=640, help="sliding window size (px)")
    parser.add_argument("--overlap", type=float, default=0.2, help="overlap between windows (0-1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video_batch(args.source, args.weights, args.window_size, args.overlap, args.output)
