import cv2
from ultralytics import YOLO
import torch
import numpy as np
from ultralytics.engine.results import Results

def dynamic_crop_and_recombine(image, detections, model_pose, padding=50):
    """
    Realiza crop dinâmico com base nas detecções e retorna os resultados ajustados para o frame original.

    Args:
        image (np.ndarray): Frame original.
        detections (list): Lista de detecções do YOLO no formato [x1, y1, x2, y2, conf, cls].
        model_pose: Modelo YOLO carregado para realizar inferência em cada crop.
        padding (int): Quantidade de padding ao redor das detecções.

    Returns:
        np.ndarray: Imagem anotada com detecções ajustadas.
    """
    h, w, _ = image.shape

    # Listas para armazenar as detecções combinadas
    combined_boxes = []
    combined_keypoints = []

    for det in detections:
        # Coordenadas do bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2, conf, cls = map(int, det[:6])

        # Adiciona padding ao redor do bounding box
        x1_crop = max(x1 - padding, 0)
        y1_crop = max(y1 - padding, 0)
        x2_crop = min(x2 + padding, w)
        y2_crop = min(y2 + padding, h)

        # Realiza o crop da imagem
        crop = image[y1_crop:y2_crop, x1_crop:x2_crop]

        # Processa o crop com YOLO novamente
        crop_results = model_pose.predict(source=crop, conf=0.2, iou=0.5,
                                          verbose=False, device=0, imgsz=128)
        crop_boxes = crop_results[0].boxes.data.cpu().numpy()
        crop_keypoints = crop_results[0].keypoints.data.cpu().numpy()

        # Verifica se há detecções
        if len(crop_boxes) == 0:
            # print("Nenhuma detecção encontrada no crop!")
            continue  # Pula este crop se não houver detecção

        # Ajusta as detecções do crop para o sistema de coordenadas original
        for i, crop_det in enumerate(crop_boxes):
            # Se a detecção tem 4 valores (caixa de limites), adiciona os valores restantes
            if len(crop_det) == 4:
                cx1, cy1, cx2, cy2 = crop_det
                cconf = 1.0  # Valor default de confiança
                ccls = 0  # Classe default
            else:
                cx1, cy1, cx2, cy2, cconf, ccls = crop_det

            # Ajusta as coordenadas
            cx1 += x1_crop
            cy1 += y1_crop
            cx2 += x1_crop
            cy2 += y1_crop
            combined_boxes.append([cx1, cy1, cx2, cy2, cconf, ccls])

            # Ajusta os keypoints
            keypoints = crop_keypoints[i]
            keypoints[:, 0] += x1_crop  # Ajusta x
            keypoints[:, 1] += y1_crop  # Ajusta y
            combined_keypoints.append(keypoints)

    # Convertendo listas para arrays numpy
    combined_boxes_array = np.array(combined_boxes, dtype=np.float32)
    combined_keypoints_array = np.array(combined_keypoints, dtype=np.float32)

    # Criando tensores diretamente dos arrays numpy
    combined_boxes_tensor = torch.tensor(combined_boxes_array, dtype=torch.float32, device='cuda:0')
    combined_keypoints_tensor = torch.tensor(combined_keypoints_array, dtype=torch.float32, device='cuda:0')

    # Garante que o tensor de caixas e keypoints não estão vazios
    if combined_boxes_tensor.shape[0] == 0:
        # print("Nenhuma detecção para ser plotada.")
        return image  # Retorna a imagem sem anotações

    results = Results(
        orig_img=image,
        boxes=combined_boxes_tensor,
        keypoints=combined_keypoints_tensor,
        names={0: 'person'},  # Substitua pelos nomes corretos, se necessário
        path=""  # Placeholder para o argumento obrigatório "path"
    )

    # Gera a imagem anotada
    annotated_image = results.plot()

    return annotated_image


def process_video(video_path, model, model_pose, output_path):
    """
    Processa um vídeo frame a frame, realizando detecção e aplicação de crop dinâmico.
    Cria um vídeo com os frames processados.

    Args:
        video_path (str): Caminho do vídeo de entrada.
        model: Modelo YOLO carregado para detecção de objetos.
        model_pose: Modelo YOLO com pose carregado para inferência nos crops.
        output_path (str): Caminho para salvar o vídeo processado.
    """
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    # Define o codec para salvar o vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cria o escritor de vídeo para salvar o vídeo processado
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_number == 800:
            break

        # Realiza predições na imagem completa
        results_full = model.predict(source=frame, conf=0.25, iou=0.5, classes=[0], imgsz=1280,
                                     verbose=False, device=0)

        # Extrai detecções iniciais
        initial_detections = results_full[0].boxes.data.cpu().numpy()

        if len(initial_detections) > 0:
            # Aplica o crop dinâmico e recombina as detecções
            annotated_frame = dynamic_crop_and_recombine(frame, initial_detections, model_pose, padding=50)
        else:
            annotated_frame = results_full[0].plot()

        # Escreve o frame processado no vídeo de saída
        out.write(annotated_frame)
        frame_number += 1
        # print("Frame number: ", frame_number, end='\r', flush=True)
        print("Frame number: ", frame_number, flush=True)


    # Libera os recursos
    cap.release()
    out.release()


if __name__ == "__main__":
    # Carrega os modelos YOLO
    model = YOLO("yolov8x.pt")
    model_pose = YOLO("yolov8n-pose.pt")

    # Caminho do vídeo de entrada e saída
    video_path = '/path/video/name.mp4'
    output_path = './output_video.mp4'

    # Processa o vídeo e cria um novo vídeo com os frames processados
    process_video(video_path, model, model_pose, output_path)
