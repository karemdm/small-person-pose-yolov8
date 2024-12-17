import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
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
    frames_batch = []
    window_batch = []  # Acumula todas as janelas para processamento em batch
    window_positions = []  # Salva as posições (x1, y1) de cada janela

    while True:
        ret, frame = cap.read()
        if not ret or frame_number == 100:
            break

        frames_batch.append(frame)
        frame_number += 1

        # Gerar janelas deslizantes para o frame atual
        for x1, y1, x2, y2 in sliding_window(frame, window_size, overlap):
            window = frame[y1:y2, x1:x2]
            window_batch.append(window)
            window_positions.append((x1, y1))

        # Processar quando atingimos o batch_size ou o final do vídeo
        if len(frames_batch) == batch_size or not ret:
            # Processar TODAS as janelas acumuladas em batch
            if window_batch:
                all_boxes = []
                all_keypoints = []

                # results = model(window_batch, device=0, verbose=False, conf=0.3, iou=0.8)
                results = model.track(window_batch, device=0, verbose=False, conf=0.3, iou=0.8)
                num = 0
                idx = 0
                idx_batch = 0

                for result in results:
                    num += 1
                    non_zero_keypoints = (result.keypoints.xy != 0).sum(dim=(1, 2))
                    if result.boxes is not None and len(result.boxes) > 0 \
                            and (non_zero_keypoints >= 2).any():
                        boxes = result.boxes.xyxy
                        keypoints = result.keypoints.xy

                        # Adicionando confiança e ID da classe
                        confidence = result.boxes.conf  # Confiança das detecções
                        class_name = result.boxes.cls # num das classes (no caso, pessoas)
                        class_id = result.boxes.id # ID do tracker

                        boxes = adjust_coordinates(boxes, window_positions[idx][0], window_positions[idx][1])
                        keypoints = adjust_keypoints(keypoints, window_positions[idx][0], window_positions[idx][1])

                        # Adicionando confiança e ID da classe
                        if class_id is not None:
                            all_boxes.append(torch.cat((boxes, class_id.unsqueeze(1), confidence.unsqueeze(1), class_name.unsqueeze(1)), dim=1))
                        else:
                            all_boxes.append(torch.cat((boxes, confidence.unsqueeze(1), class_name.unsqueeze(1)), dim=1))
                        all_keypoints.append(keypoints)

                    if num == (len(window_positions) // batch_size):

                        # Verifique se all_boxes foi corretamente concatenado para evitar erros
                        if len(all_boxes) > 0:
                            combined_boxes, combined_keypoints = merge_detections(all_boxes, all_keypoints)

                            if combined_boxes is not None and combined_boxes.size(0) > 0 and \
                                combined_keypoints is not None and not (combined_keypoints == 0).all():
                                results = Results(
                                    orig_img=frames_batch[idx_batch],
                                    boxes=combined_boxes,
                                    keypoints=combined_keypoints,
                                    names={0: "person"},
                                    path=""
                                )
                                plotted_frame = results.plot()
                            else:
                                plotted_frame = frames_batch[idx_batch]
                        else:
                            plotted_frame = frames_batch[idx_batch]

                        num = 0
                        idx_batch += 1
                        out.write(plotted_frame)
                        # frame_number += 1
                        print(f"Processed frame: {frame_number}", end="\r")
                    idx += 1

            # Limpar batches
            frames_batch = []
            window_batch = []
            window_positions = []

    print("\nProcessing complete.")
    cap.release()
    out.release()


# Configurações
video_path = "./client/client - Video obra.mp4"
model_path = "yolov8n-pose.pt"  # Substitua pelo caminho do modelo treinado
output_path = "processed_video.mp4"
window_size = 640
overlap = 0.2

process_video_batch(video_path, model_path, window_size, overlap, output_path)
