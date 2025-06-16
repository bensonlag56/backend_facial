import os
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image
import numpy as np

def recognize_user(image_path, usuarios, threshold=0.8):
    """
    Compara una imagen contra todos los modelos CNN de usuarios, incluyendo 'no_rostro'.
    Retorna el usuario con la mayor confianza si supera el umbral, o None si no hay coincidencias fuertes.
    """
    img = preprocess_image(image_path)
    mejor_usuario = None
    mejor_confianza = 0.0
    clase_detectada = None

    for usuario in usuarios:
        model_path = usuario['ruta_modelo']
        if os.path.exists(model_path):
            model = load_model(model_path)
            prediction = model.predict(img)

            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            if confidence > mejor_confianza:
                mejor_confianza = confidence
                mejor_usuario = usuario
                clase_detectada = 'rostro' if predicted_class == 0 else 'no_rostro'

    if mejor_usuario and mejor_confianza > threshold:
        return {
            'codigo_unico': mejor_usuario['codigo_unico'],
            'nombre': mejor_usuario['nombre'],
            'apellido': mejor_usuario['apellido'],
            'requisitoriado': mejor_usuario['requisitoriado'],
            'confidence': float(mejor_confianza),
            'clase_detectada': clase_detectada
        }
    return None
