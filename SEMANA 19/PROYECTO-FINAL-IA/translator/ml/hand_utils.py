import numpy as np
import math
import cv2
from tensorflow.keras.models import load_model
import os
import mediapipe as mp

# ---------------------------
# Carga única del modelo
# ---------------------------
from .model_loader import get_model
model = get_model()     

# Mapeo de índices a letras (A‑Z)
IDX2CLASS = [
  'A','B','C','D','E','F','G','H','I','J','K','L','M',
  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

CLASS_MAP = {i: name for i, name in enumerate(IDX2CLASS)}

# Target size usado en entrenamiento
TARGET_SIZE = (256,256) 

# ---------------------------
# Inicializar Mediapipe una sola vez
# ---------------------------
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,   # procesamos frame a frame
    max_num_hands=1,
    min_detection_confidence=0.5,
)

def crop_hand(img_bgr: np.ndarray) -> np.ndarray | None:
    """Recorta la región de la mano usando Mediapipe.
    Devuelve ROI en BGR o None si no se detecta mano."""
    h, w, _ = img_bgr.shape
    results = hands_detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None

    # Coordenadas normalizadas de la mano
    xs = [lm.x for lm in results.multi_hand_landmarks[0].landmark]
    ys = [lm.y for lm in results.multi_hand_landmarks[0].landmark]
    x_min, x_max = max(min(xs) - 0.05, 0), min(max(xs) + 0.05, 1)
    y_min, y_max = max(min(ys) - 0.05, 0), min(max(ys) + 0.05, 1)

    # Convertir a píxeles y recortar
    x_min, x_max = int(x_min * w), int(x_max * w)
    y_min, y_max = int(y_min * h), int(y_max * h)
    if x_max - x_min < 10 or y_max - y_min < 10:
        return None  # recorte demasiado pequeño / ruido

    return img_bgr[y_min:y_max, x_min:x_max]

def square_pad(img, size=TARGET_SIZE[0]):
    """
    Ajusta el recorte de la mano a un lienzo cuadrado de fondo blanco (size×size px),
    manteniendo la proporción original sin distorsionar la imagen.
    """
    # Obtener dimensiones de la imagen original
    h, w = img.shape[:2]
    # Crear lienzo blanco (RGB) de tamaño size×size
    bg = np.zeros((size, size, 3), np.uint8)
    if h > w:
        # Caso imagen “vertical”: la altura es mayor que el ancho
        # 1. Calcular factor de escala para que la altura = size
        k = size / h
        # 2. Calcular nuevo ancho proporcional
        w_new = math.ceil(w * k)
        # 3. Redimensionar manteniendo la relación de aspecto
        img_r = cv2.resize(img, (w_new, size))
        # 4. Calcular desplazamiento horizontal para centrar
        x_gap = (size - w_new)//2
        # 5. Pegar la imagen redimensionada en el lienzo
        bg[:, x_gap:x_gap+w_new] = img_r
    else:
        # Caso imagen “horizontal” o cuadrada: el ancho ≥ altura
        # 1. Calcular factor de escala para que el ancho = size
        k = size / w
        # 2. Calcular nueva altura proporcional
        h_new = math.ceil(h * k)
        # 3. Redimensionar manteniendo la relación de aspecto
        img_r = cv2.resize(img, (size, h_new))
        # 4. Calcular desplazamiento vertical para centrar
        y_gap = (size - h_new)//2
        # 5. Pegar la imagen redimensionada en el lienzo
        bg[y_gap:y_gap+h_new, :] = img_r
         # Devolver el lienzo cuadrado con la imagen centrada
    return bg

def preprocess_frame(img_bgr: np.ndarray) -> np.ndarray:
    """Convierte BGR→RGB, resize a TARGET_SIZE, normaliza [0‑1] y expande dims."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, TARGET_SIZE)
    img_norm = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_norm, axis=0)


def predict_from_frame(img_bgr: np.ndarray) -> str:
    """Devuelve la letra ASL predicha para el frame completo de la webcam."""
    roi = crop_hand(img_bgr)
    if roi is None:
        return 'mano no detectada', 0.0 # no mano detectada
    
    
    roi = square_pad(roi, 150)
    x = preprocess_frame(roi)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    conf = float(preds[0][idx]);
    return CLASS_MAP[idx], conf