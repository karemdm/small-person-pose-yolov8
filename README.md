# Video Processing and Dynamic Cropping with YOLOv8

This repository contains Python scripts for processing videos and performing dynamic cropping of images with YOLOv8 for object detection and pose estimation.

## Script 1: Video Processing with Sliding Windows

### Overview

This script processes a video using YOLOv8 Pose model in a sliding window approach. It captures frames from the video, applies the model to detect objects (people) and keypoints, and then merges overlapping detections using the Intersection Over Union (IOU) metric. Finally, the processed frames are saved to a new video file.

### Key Functions

#### `sliding_window(image, window_size, overlap)`
Generates sliding window coordinates for the image, where each window will be analyzed separately. The step size is calculated based on the window size and overlap percentage.

**Arguments:**
- `image`: The input image (frame from the video).
- `window_size`: The size of the sliding window.
- `overlap`: The percentage of overlap between consecutive windows.

#### `adjust_coordinates(detections, x_offset, y_offset)`
Adjusts the coordinates of detected bounding boxes for the original image dimensions.

**Arguments:**
- `detections`: Tensor of detected boxes.
- `x_offset`, `y_offset`: Offsets based on window position.

#### `adjust_keypoints(keypoints, x_offset, y_offset)`
Adjusts the coordinates of detected keypoints for the original image dimensions.

**Arguments:**
- `keypoints`: Tensor of detected keypoints.
- `x_offset`, `y_offset`: Offsets based on window position.

#### `merge_detections(all_boxes, all_keypoints, iou_threshold=0.8)`
Merges overlapping detections based on IOU (Intersection over Union). It discards redundant detections and keeps the one with the higher confidence.

**Arguments:**
- `all_boxes`: List of all bounding boxes.
- `all_keypoints`: List of all detected keypoints.
- `iou_threshold`: IOU threshold for merging overlapping detections.

#### `process_video(video_path, model_path, window_size, overlap, output_path)`
Processes the video by applying the YOLOv8 Pose model on each frame using sliding windows, merging overlapping detections, and saving the processed frames into a new video.

**Arguments:**
- `video_path`: Path to the input video.
- `model_path`: Path to the YOLOv8 Pose model.
- `window_size`: Size of the sliding window.
- `overlap`: Percentage of overlap between sliding windows.
- `output_path`: Path to save the processed video.

### Example Usage

```python
video_path = "/path/to/video.mp4"
model_path = "yolov8m-pose.pt"  # Replace with your trained model path
output_path = "processed_video.mp4"
window_size = 640
overlap = 0.2

process_video(video_path, model_path, window_size, overlap, output_path)

---

# Script 2: Dynamic Cropping and Pose Estimation

Este projeto processa vídeos quadro a quadro, usando o modelo YOLOv8 para detecção de objetos e o modelo YOLOv8 Pose para detectar e ajustar os pontos de pose. O vídeo de entrada é processado cortando dinamicamente as regiões ao redor das detecções e recombinando-as com base nas coordenadas originais do quadro.

## Funcionalidade

O projeto implementa os seguintes recursos:

- **Detecção de Objetos:** Usa o modelo YOLOv8 para detectar objetos nos quadros do vídeo.
- **Estimativa de Pose:** Após a detecção de objetos, o modelo YOLOv8 Pose é usado para detectar os pontos de pose e realizar o corte dinâmico ao redor dessas áreas.
- **Recombinação de Detecções:** Combina os resultados dos cortes e ajusta-os de volta para o sistema de coordenadas original do quadro.
- **Processamento de Vídeo:** O vídeo é processado quadro a quadro, com as detecções e cortes dinâmicos aplicados, e os quadros anotados são salvos em um novo arquivo de vídeo.

## Estrutura do Código

### `dynamic_crop_and_recombine(image, detections, model_pose, padding=50)`

Realiza o corte dinâmico com base nas detecções de objetos, ajusta as coordenadas dos cortes para o sistema de coordenadas do quadro original e recombina as detecções.

### `process_video(video_path, model, model_pose, output_path)`

Processa o vídeo quadro a quadro, realiza as detecções e cortes dinâmicos, e cria um vídeo com os quadros anotados.

## Requisitos

Este projeto requer os seguintes pacotes:

- `cv2` (OpenCV)
- `ultralytics` (para YOLOv8)
- `torch` (PyTorch)
- `numpy`

Você pode instalar as dependências usando o seguinte comando:

```bash
pip install opencv-python ultralytics torch numpy
```

### How to Use

#### Loading YOLO Models

The project uses two YOLOv8 models: `yolov8x.pt` for object detection and `yolov8n-pose.pt` for pose estimation.

#### Processing the Video

Define the paths for the input and output video files in the code. The input video will be processed, and the annotated video will be saved at the output path.

##### Example usage:

```python
model = YOLO("yolov8x.pt")  # YOLO object detection model
model_pose = YOLO("yolov8n-pose.pt")  # YOLO pose model

video_path = '/path/to/input_video.mp4'  # Path to the input video
output_path = '/path/to/output_video.mp4'  # Path to the output video

process_video(video_path, model, model_pose, output_path)
```

##### Explanation of Parameters
`video_path`: The path to the input video.
`model`: The YOLO model used for object detection.
`model_pose`: The YOLO model used for pose estimation.
`output_path`: The path where the processed video will be saved.
Notes
The code processes the video starting from frame 500 to frame 800. You can adjust these values in the process_video() function by modifying cap.set(cv2.CAP_PROP_POS_FRAMES, 500) and the loop condition.
The padding around each detection can be adjusted by changing the padding parameter in the dynamic_crop_and_recombine() function.


