# app.py
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import cv2
import sqlite3
import os
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Configuración
DATABASE = 'users.db'
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
}
LBP_PARAMS = {
    'P': 8,
    'R': 1,
    'method': 'uniform'
}
SIFT_PARAMS = {
    'nfeatures': 50,
    'nOctaveLayers': 3,
    'contrastThreshold': 0.04,
    'edgeThreshold': 10,
    'sigma': 1.6
}
MATCH_THRESHOLD = 0.7  # Umbral de similitud

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                apellido TEXT NOT NULL,
                codigo_unico TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                requisitoriado BOOLEAN NOT NULL,
                imagen_facial TEXT,
                hog_features BLOB,
                lbp_features BLOB,
                sift_features BLOB
            )
        ''')
        conn.commit()

def image_to_array(image_data):
    img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    return np.array(img)

def augment_image(image):
    augmented_images = []

    # Original
    augmented_images.append(image)

    # Flip horizontal
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotate +10°
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), 10, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    augmented_images.append(rotated)

    # Rotate -10°
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), -10, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    augmented_images.append(rotated)

    # Add Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented_images.append(noisy)

    return augmented_images

def extract_features(image_array):
    # Redimensionar imagen a tamaño fijo (ejemplo: 128x128)
    fixed_size = (128, 128)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, fixed_size)

    # HOG
    hog_features = hog(
        gray,
        orientations=HOG_PARAMS['orientations'],
        pixels_per_cell=HOG_PARAMS['pixels_per_cell'],
        cells_per_block=HOG_PARAMS['cells_per_block'],
        block_norm=HOG_PARAMS['block_norm'],
        transform_sqrt=HOG_PARAMS['transform_sqrt']
    )

    # LBP
    lbp = local_binary_pattern(
        gray,
        P=LBP_PARAMS['P'],
        R=LBP_PARAMS['R'],
        method=LBP_PARAMS['method']
    )
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalización

    # SIFT
    sift = cv2.SIFT_create(
        nfeatures=SIFT_PARAMS['nfeatures'],
        nOctaveLayers=SIFT_PARAMS['nOctaveLayers'],
        contrastThreshold=SIFT_PARAMS['contrastThreshold'],
        edgeThreshold=SIFT_PARAMS['edgeThreshold'],
        sigma=SIFT_PARAMS['sigma']
    )
    _, sift_descriptors = sift.detectAndCompute(gray, None)

    sift_features = np.mean(sift_descriptors, axis=0) if sift_descriptors is not None and len(sift_descriptors) > 0 else np.zeros(128)

    return {
        'hog': hog_features.tobytes(),
        'lbp': lbp_hist.tobytes(),
        'sift': sift_features.tobytes()
    }

def compare_features(query_features, stored_features):
    # Convertir bytes a numpy arrays
    query_hog = np.frombuffer(query_features['hog'], dtype=np.float64)
    stored_hog = np.frombuffer(stored_features['hog'], dtype=np.float64)
    
    query_lbp = np.frombuffer(query_features['lbp'], dtype=np.float32)
    stored_lbp = np.frombuffer(stored_features['lbp'], dtype=np.float32)
    
    query_sift = np.frombuffer(query_features['sift'], dtype=np.float32)
    stored_sift = np.frombuffer(stored_features['sift'], dtype=np.float32)
    
    # Calcular similitudes (1 es perfecto, 0 es nada similar)
    hog_sim = cosine_similarity([query_hog], [stored_hog])[0][0]
    lbp_sim = cosine_similarity([query_lbp], [stored_lbp])[0][0]
    
    # Manejar caso donde no hay features SIFT
    if query_sift.size == 0 or stored_sift.size == 0:
        sift_sim = 0
    else:
        sift_sim = cosine_similarity([query_sift], [stored_sift])[0][0]
    
    # Ponderación de características (ajustable)
    weights = {'hog': 0.4, 'lbp': 0.3, 'sift': 0.3}
    total_sim = (hog_sim * weights['hog'] + 
                lbp_sim * weights['lbp'] + 
                sift_sim * weights['sift'])
    
    return total_sim

@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.json
        required_fields = ['nombre', 'apellido', 'codigo_unico', 'email', 'requisitoriado', 'imagen_facial']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Faltan campos requeridos'}), 400
        
        # Validar imagen
        try:
            img_array = image_to_array(data['imagen_facial'])
        except:
            return jsonify({'error': 'Imagen inválida'}), 400
        
        # Aplicar aumento de datos
        augmented_images = augment_image(img_array)

        # Extraer características de todas las imágenes aumentadas
        features_list = [extract_features(img) for img in augmented_images]

        # Promediar características
        avg_hog = np.mean([np.frombuffer(f['hog'], dtype=np.float64) for f in features_list], axis=0)
        avg_lbp = np.mean([np.frombuffer(f['lbp'], dtype=np.float32) for f in features_list], axis=0)
        avg_sift = np.mean([np.frombuffer(f['sift'], dtype=np.float32) for f in features_list], axis=0)

        # Convertir de nuevo a bytes
        final_features = {
            'hog': avg_hog.tobytes(),
            'lbp': avg_lbp.tobytes(),
            'sift': avg_sift.tobytes()
        }
        
        # Guardar en base de datos
        with get_db() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users 
                    (nombre, apellido, codigo_unico, email, requisitoriado, imagen_facial, hog_features, lbp_features, sift_features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['nombre'],
                    data['apellido'],
                    data['codigo_unico'],
                    data['email'],
                    bool(data['requisitoriado']),
                    data['imagen_facial'],
                    final_features['hog'],
                    final_features['lbp'],
                    final_features['sift']
                ))
                conn.commit()
            except sqlite3.IntegrityError as e:
                return jsonify({'error': 'El código único o email ya existen'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        return jsonify({'message': 'Usuario registrado exitosamente'}), 201
    except Exception as e:
        return jsonify({'error': f'Error en el servidor: {str(e)}'}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        if not request.json or 'imagen' not in request.json:
            return jsonify({'error': 'Se requiere una imagen'}), 400

        try:
            img_array = image_to_array(request.json['imagen'])
        except Exception as e:
            return jsonify({'error': f'Imagen inválida: {str(e)}'}), 400

        if img_array is None:
            return jsonify({'error': 'No se pudo procesar la imagen'}), 400

        query_features = extract_features(img_array)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, nombre, apellido, codigo_unico, email, requisitoriado, 
                       hog_features, lbp_features, sift_features 
                FROM users
            ''')
            users = cursor.fetchall()

        if not users:
            return jsonify({'match': False}), 200

        matches = []
        for user in users:
            stored_features = {
                'hog': user['hog_features'],
                'lbp': user['lbp_features'],
                'sift': user['sift_features']
            }

            similarity = compare_features(query_features, stored_features)

            if similarity >= MATCH_THRESHOLD:
                matches.append({
                    'id': user['id'],
                    'nombre': user['nombre'],
                    'apellido': user['apellido'],
                    'codigo_unico': user['codigo_unico'],
                    'email': user['email'],
                    'requisitoriado': bool(user['requisitoriado']),
                    'confidence': float(similarity)
                })

        if matches:
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            best_match = matches[0]

            if best_match['requisitoriado']:
                print("¡ALERTA DE SEGURIDAD! Usuario Requisitoriado Detectado. Notificación Enviada a la Policía")

            return jsonify({
                'match': True,
                'user': best_match,
                'all_matches': matches,
                'features_used': {
                    'hog': 'Histogram of Oriented Gradients',
                    'lbp': 'Local Binary Patterns',
                    'sift': 'Scale-Invariant Feature Transform'
                }
            }), 200
        else:
            return jsonify({'match': False}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error en el servidor: {str(e)}'}), 500

@app.route('/users', methods=['GET'])
def get_users():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, nombre, apellido, codigo_unico, email, requisitoriado 
                FROM users
            ''')
            users = cursor.fetchall()
        
        users_list = [dict(user) for user in users]
        for user in users_list:
            user['requisitoriado'] = bool(user['requisitoriado'])
        
        return jsonify(users_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({'error': 'Usuario no encontrado'}), 404
        
        return jsonify({'message': 'Usuario eliminado exitosamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)