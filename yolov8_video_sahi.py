import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
# from torchvision.ops import box_iou
import torchvision.ops as ops


def sliding_window(image, window_size, overlap):
    """
    Gera coordenadas de janelas deslizantes na imagem,
    garantindo que os pixels das bordas também sejam processados.
    """
    step = int(window_size * (1 - overlap))
    h, w, _ = image.shape

    for y in range(0, h, step):
        for x in range(0, w, step):
            x1 = x
            y1 = y
            x2 = min(x + window_size, w)  # Garante que a janela não exceda a largura da imagem
            y2 = min(y + window_size, h)  # Garante que a janela não exceda a altura da imagem
            yield x1, y1, x2, y2


def adjust_coordinates(detections, x_offset, y_offset):
    """
    Ajusta as coordenadas das detecções para as dimensões originais.
    """
    if detections is None or detections.size(0) == 0:
        return detections

    detections = detections.clone()
    detections[:, [0, 2]] += x_offset  # Ajusta x_min e x_max
    detections[:, [1, 3]] += y_offset  # Ajusta y_min e y_max
    return detections


def adjust_keypoints(keypoints, x_offset, y_offset):
    """
    Ajusta as coordenadas dos keypoints para as dimensões originais,
    ignorando valores de 0 (em x e y).
    """
    if keypoints is None or keypoints.size(0) == 0:
        return keypoints

    keypoints = keypoints.clone()

    # Somar o offset apenas onde os valores de keypoints não são 0
    keypoints[:, :, 0] = torch.where(keypoints[:, :, 0] != 0, keypoints[:, :, 0] + x_offset, keypoints[:, :, 0])
    keypoints[:, :, 1] = torch.where(keypoints[:, :, 1] != 0, keypoints[:, :, 1] + y_offset, keypoints[:, :, 1])

    return keypoints


def merge_detections(all_boxes, all_keypoints, iou_threshold=0.5):
    target_device = all_boxes[0].device if len(all_boxes) > 0 else 'cpu'

    # Move boxes e keypoints para o mesmo dispositivo
    all_boxes = [box.to(target_device) for box in all_boxes]
    all_keypoints = [keypoint.to(target_device) for keypoint in all_keypoints]

    # Concatena as boxes e keypoints
    combined_boxes = torch.cat(all_boxes, dim=0) if len(all_boxes) > 0 else torch.empty(0, device=target_device)
    combined_keypoints = torch.cat(all_keypoints, dim=0) if len(all_keypoints) > 0 else torch.empty(0, device=target_device)

    # Certifique-se de que a caixa tenha apenas 4 colunas: [x1, y1, x2, y2]
    # Se a caixa tiver mais de 4 colunas, vamos considerar apenas as primeiras 4 colunas para NMS.
    boxes = combined_boxes[:, :4]  # Seleciona apenas as 4 primeiras colunas (coordenadas da caixa)
    scores = combined_boxes[:, 5]  # Assume que a 5ª coluna é a confiança

    # Filtragem por IoU (remove boxes com alta sobreposição)
    keep = ops.nms(boxes, scores, iou_threshold)
    filtered_boxes = combined_boxes[keep]
    filtered_keypoints = combined_keypoints[keep]

    return filtered_boxes, filtered_keypoints


def process_video(video_path, model_path, window_size, overlap, output_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

            non_zero_keypoints = (results[0].keypoints.xy != 0).sum(dim=(1, 2))
            if results[0].boxes is not None and len(results[0].boxes) > 0 and (non_zero_keypoints >= 2).any():
                boxes = results[0].boxes.xyxy
                keypoints = results[0].keypoints.xy

                # Adicionando confiança e ID da classe
                confidence = results[0].boxes.conf  # Confiança das detecções
                class_name = results[0].boxes.cls # num das classes (no caso, pessoas)
                class_id = results[0].boxes.id # ID do tracker

                boxes = adjust_coordinates(boxes, x1, y1)
                keypoints = adjust_keypoints(keypoints, x1, y1)

                # Adicionando confiança e ID da classe
                all_boxes.append(torch.cat((boxes, class_id.unsqueeze(1), confidence.unsqueeze(1), class_name.unsqueeze(1)), dim=1))
                all_keypoints.append(keypoints)

        # Verifique se all_boxes foi corretamente concatenado para evitar erros
        if len(all_boxes) > 0:
            combined_boxes, combined_keypoints = merge_detections(all_boxes, all_keypoints)

            if combined_boxes is not None and combined_boxes.size(0) > 0 and \
                combined_keypoints is not None and not (combined_keypoints == 0).all():
                results = Results(
                    orig_img=frame,
                    boxes=combined_boxes,
                    keypoints=combined_keypoints,
                    names={0: "person"},
                    path=""
                )
                plotted_frame = results.plot()
            else:
                plotted_frame = frame
        else:
            plotted_frame = frame

        out.write(plotted_frame)
        frame_number += 1
        print(f"Processed frame: {frame_number}", end="\r")

    print(f"Processed frame: {frame_number}")
    cap.release()
    out.release()
    print("\nProcessing complete.")


def process_video_batch(video_path, model_path, window_size, overlap, output_path, batch_size=16):
    """
    Processa o vídeo aplicando o modelo YOLOv8-Pose em janelas deslizantes usando batches.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

    # Armazenar múltiplos frames para processamento em batch
    frames_batch = []
    frame_indices = []

    while True:
        ret, frame = cap.read()
        if not ret or frame_number == 100:
            break

        frames_batch.append(frame)
        frame_indices.append(frame_number)
        frame_number += 1

        # Processar quando atingimos o batch_size ou quando o vídeo termina
        if len(frames_batch) == batch_size or not ret:
            all_detections = []  # Para armazenar detecções de todos os frames no batch

            for frame_idx, frame in enumerate(frames_batch):
                all_boxes = []
                all_keypoints = []

                for x1, y1, x2, y2 in sliding_window(frame, window_size, overlap):
                    window = frame[y1:y2, x1:x2]
                    # results = model.predict(window, device=0, verbose=False, conf=0.3, iou=0.8)
                    results = model.track(window, device=0, verbose=False, conf=0.3, iou=0.8)

                    non_zero_keypoints = (results[0].keypoints.xy != 0).sum(dim=(1, 2))
                    if results[0].boxes is not None and len(results[0].boxes) > 0 \
                            and (non_zero_keypoints >= 2).any():
                        boxes = results[0].boxes.xyxy
                        keypoints = results[0].keypoints.xy

                        # Adicionando confiança e ID da classe
                        confidence = results[0].boxes.conf  # Confiança das detecções
                        class_name = results[0].boxes.cls # num das classes (no caso, pessoas)
                        class_id = results[0].boxes.id # ID do tracker

                        boxes = adjust_coordinates(boxes, x1, y1)
                        keypoints = adjust_keypoints(keypoints, x1, y1)

                        # Adicionando confiança e ID da classe
                        all_boxes.append(torch.cat((boxes, class_id.unsqueeze(1), confidence.unsqueeze(1), class_name.unsqueeze(1)), dim=1))
                        all_keypoints.append(keypoints)

                combined_boxes, combined_keypoints = merge_detections(all_boxes, all_keypoints)
                all_detections.append((combined_boxes, combined_keypoints))

            # Após coletar as detecções do batch, renderizar os frames
            for frame_idx, (combined_boxes, combined_keypoints) in enumerate(all_detections):
                frame = frames_batch[frame_idx]

                if combined_boxes is not None and combined_boxes.size(0) > 0 and \
                    combined_keypoints is not None and not (combined_keypoints == 0).all():
                    results = Results(
                        orig_img=frame,
                        boxes=combined_boxes,
                        keypoints=combined_keypoints,
                        names={0: "person"},
                        path=""
                    )
                    plotted_frame = results.plot()
                else:
                    plotted_frame = frame

                out.write(plotted_frame)
                print(f"Processed frame: {frame_indices[frame_idx]}", end="\r")

            # Limpar o batch
            frames_batch = []
            frame_indices = []

    print("\nProcessing complete.")
    cap.release()
    out.release()


# Configurações
video_path = "/path/video/name.mp4"
model_path = "yolov8n-pose.pt"  # Substitua pelo caminho do modelo treinado
output_path = "processed_video.mp4"
window_size = 640
overlap = 0.2

process_video(video_path, model_path, window_size, overlap, output_path)
# process_video_batch(video_path, model_path, window_size, overlap, output_path)

