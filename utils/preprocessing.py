
import cv2
import numpy as np
from config import IMG_HEIGHT, IMG_WIDTH

def preprocess_image(image_path):
    """
    Lee una imagen desde el disco, la redimensiona al tamaño del modelo,
    la normaliza y la ajusta para predicción con la CNN.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen desde la ruta: {image_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype('float32') / 255.0  # Normalización entre 0 y 1
    img = np.expand_dims(img, axis=0)    # Ajuste a shape (1, IMG_HEIGHT, IMG_WIDTH, 3)
    return img