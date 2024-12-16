import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.utils.files import increment_path


def run(weights='yolov8n-pose.pt', source='test.mp4', view_img=False, save_img=False, exist_ok=False):
    """
    Run pose detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    yolov8_model_path = f'models/{weights}'
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=0.3,
                                                         device=0)

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # Output setup
    save_dir = increment_path(Path('ultralytics_results_with_pose') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, 500)
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success and frame_number == 1000:
            break

        # Perform pose detection
        results = get_sliced_prediction(frame,
                                        detection_model,
                                        slice_height=512,
                                        slice_width=512,
                                        overlap_height_ratio=0.2,
                                        overlap_width_ratio=0.2)
        keypoint_prediction_list = results.keypoint_prediction_list

        # Loop over detected keypoints
        for keypoints in keypoint_prediction_list:
            for keypoint in keypoints.keypoints:
                x, y, confidence = keypoint
                if confidence > 0.3:  # Threshold for keypoint confidence
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Optionally draw lines between keypoints (e.g., between specific body parts)
        # Example: Draw lines between specific body parts (e.g., shoulders, elbows, etc.)
        for keypoints in keypoint_prediction_list:
            # Draw lines connecting body parts (if desired)
            for pair in keypoints.get_keypoint_pairs():
                x1, y1 = pair[0]
                x2, y2 = pair[1]
                if keypoints.get_confidence(pair[0]) > 0.3 and keypoints.get_confidence(pair[1]) > 0.3:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        if view_img:
            cv2.imshow(Path(source).stem, frame)
        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_number += 1
        print(f"Processed frame: {frame_number}", end="\r")

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-pose.pt', help='initial weights path')
    parser.add_argument('--source', type=str, required=True, help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()


def main():
    # Definindo os parâmetros diretamente
    opt = {
        'weights': 'yolov8n-pose.pt',  # Caminho do modelo
        'source': '/path/to/input_video.mp4',
        'view_img': False,  # Mostrar resultados
        'save_img': True,  # Salvar resultados
        'exist_ok': True  # Permitir sobrescrever resultados existentes
    }

    # Chama a função run com os parâmetros definidos
    run(**opt)

if __name__ == '__main__':
    main()
