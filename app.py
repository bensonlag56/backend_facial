# app.py
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
import json
from scipy.signal import convolve2d
import psycopg2
import psycopg2.extras

app = Flask(__name__)

# Configuración
DATABASE_URL = os.getenv('DATABASE_URL')
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
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                nombre TEXT NOT NULL,
                apellido TEXT NOT NULL,
                codigo_unico TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                requisitoriado BOOLEAN NOT NULL,
                imagen_facial TEXT,
                hog_features BYTEA,
                lbp_features BYTEA,
                sift_features BYTEA,
                lpq_features BYTEA
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

def lpq(image, win_size=3):
    rho = 0.90
    STFTalpha = 1.0 / win_size
    conv_mode = 'valid'

    x = np.arange(-win_size // 2 + 1., win_size // 2 + 1.)
    n = len(x)
    w0 = np.ones(n)
    w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)

    filter_resp = []
    filter_resp.append(convolve2d(image, np.real(w1[:, np.newaxis] * w0), mode=conv_mode))
    filter_resp.append(convolve2d(image, np.imag(w1[:, np.newaxis] * w0), mode=conv_mode))
    filter_resp.append(convolve2d(image, np.real(w0[:, np.newaxis] * w1), mode=conv_mode))
    filter_resp.append(convolve2d(image, np.imag(w0[:, np.newaxis] * w1), mode=conv_mode))

    freq_resp = np.stack(filter_resp, axis=-1)
    lpq_desc = ((freq_resp > 0) * (1 << np.arange(freq_resp.shape[-1]))).sum(axis=-1).flatten()

    hist, _ = np.histogram(lpq_desc, bins=256, range=(0, 255))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)

    if np.isnan(hist).any():
        hist = np.zeros_like(hist)

    return hist

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

    # LPQ
    lpq_features = lpq(gray)

    return {
        'hog': hog_features.tobytes(),
        'lbp': lbp_hist.tobytes(),
        'sift': sift_features.tobytes(),
        'lpq': lpq_features.tobytes()
    }

def compare_features(query_features, stored_features):
    # Convertir bytes a numpy arrays
    query_hog = np.frombuffer(query_features['hog'], dtype=np.float64)
    stored_hog = np.frombuffer(stored_features['hog'], dtype=np.float64)
    
    query_lbp = np.frombuffer(query_features['lbp'], dtype=np.float32)
    stored_lbp = np.frombuffer(stored_features['lbp'], dtype=np.float32)
    
    query_sift = np.frombuffer(query_features['sift'], dtype=np.float32)
    stored_sift = np.frombuffer(stored_features['sift'], dtype=np.float32)

    expected_lpq_size = 256  # Adjust this based on your LPQ histogram size

    query_lpq = np.frombuffer(query_features['lpq'], dtype=np.float32)
    stored_lpq_data = stored_features['lpq']
    if stored_lpq_data is None:
        stored_lpq = np.zeros(expected_lpq_size, dtype=np.float32)
    else:
        stored_lpq = np.frombuffer(stored_lpq_data, dtype=np.float32)
    
    # Calcular similitudes (1 es perfecto, 0 es nada similar)
    hog_sim = cosine_similarity([query_hog], [stored_hog])[0][0]
    lbp_sim = cosine_similarity([query_lbp], [stored_lbp])[0][0]
    
    # Manejar caso donde no hay features SIFT
    if query_sift.size == 0 or stored_sift.size == 0:
        sift_sim = 0
    else:
        sift_sim = cosine_similarity([query_sift], [stored_sift])[0][0]

    lpq_sim = cosine_similarity([query_lpq], [stored_lpq])[0][0]
    
    # Ponderación de características (ajustable)
    weights = {'hog': 0.3, 'lbp': 0.2, 'sift': 0.3, 'lpq': 0.2}
    total_sim = (hog_sim * weights['hog'] + 
                lbp_sim * weights['lbp'] + 
                sift_sim * weights['sift'] + 
                lpq_sim * weights['lpq'])
    
    return total_sim

@app.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.json
        required_fields = ['nombre', 'apellido', 'codigo_unico', 'email', 'requisitoriado', 'imagen_frontal', 'imagen_izquierda', 'imagen_derecha']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Faltan campos requeridos'}), 400
        
        # Validar imágenes
        try:
            img_front = image_to_array(data['imagen_frontal'])
            img_left = image_to_array(data['imagen_izquierda'])
            img_right = image_to_array(data['imagen_derecha'])
        except:
            return jsonify({'error': 'Una o más imágenes son inválidas'}), 400

        # Extraer características de cada imagen
        features_front = extract_features(img_front)
        features_left = extract_features(img_left)
        features_right = extract_features(img_right)

        # Promediar características
        avg_hog = np.mean([
            np.frombuffer(features_front['hog'], dtype=np.float64),
            np.frombuffer(features_left['hog'], dtype=np.float64),
            np.frombuffer(features_right['hog'], dtype=np.float64)
        ], axis=0)

        avg_lbp = np.mean([
            np.frombuffer(features_front['lbp'], dtype=np.float32),
            np.frombuffer(features_left['lbp'], dtype=np.float32),
            np.frombuffer(features_right['lbp'], dtype=np.float32)
        ], axis=0)

        avg_sift = np.mean([
            np.frombuffer(features_front['sift'], dtype=np.float32),
            np.frombuffer(features_left['sift'], dtype=np.float32),
            np.frombuffer(features_right['sift'], dtype=np.float32)
        ], axis=0)

        avg_lpq = np.mean([
            np.frombuffer(features_front['lpq'], dtype=np.float32),
            np.frombuffer(features_left['lpq'], dtype=np.float32),
            np.frombuffer(features_right['lpq'], dtype=np.float32)
        ], axis=0)

        if np.isnan(avg_lpq).any():
            avg_lpq = np.zeros_like(avg_lpq)

        final_features = {
            'hog': avg_hog.tobytes(),
            'lbp': avg_lbp.tobytes(),
            'sift': avg_sift.tobytes(),
            'lpq': avg_lpq.tobytes()
        }
        
        # Guardar en base de datos
        with get_db() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users 
                    (nombre, apellido, codigo_unico, email, requisitoriado, imagen_facial, hog_features, lbp_features, sift_features, lpq_features)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    data['nombre'],
                    data['apellido'],
                    data['codigo_unico'],
                    data['email'],
                    bool(data['requisitoriado']),
                    None,
                    final_features['hog'],
                    final_features['lbp'],
                    final_features['sift'],
                    final_features['lpq']
                ))
                conn.commit()
            except psycopg2.IntegrityError as e:
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
                       hog_features, lbp_features, sift_features, lpq_features
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
                'sift': user['sift_features'],
                'lpq': user['lpq_features']
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
                    'sift': 'Scale-Invariant Feature Transform',
                    'lpq': 'Local Phase Quantization'
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
        nombre = request.args.get('nombre')
        apellido = request.args.get('apellido')
        codigo_unico = request.args.get('codigo_unico')

        query = 'SELECT id, nombre, apellido, codigo_unico, email, requisitoriado FROM users WHERE 1=1'
        params = []

        if nombre:
            query += ' AND nombre ILIKE %s'
            params.append(f'%{nombre}%')
        if apellido:
            query += ' AND apellido ILIKE %s'
            params.append(f'%{apellido}%')
        if codigo_unico:
            query += ' AND codigo_unico = %s'
            params.append(codigo_unico)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
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
            cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return jsonify({'error': 'Usuario no encontrado'}), 404
        
        return jsonify({'message': 'Usuario eliminado exitosamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        data = request.json
        fields = ['nombre', 'apellido', 'codigo_unico', 'email', 'requisitoriado']
        updates = []
        params = []

        for field in fields:
            if field in data:
                updates.append(f"{field} = %s")
                params.append(data[field])

        if not updates:
            return jsonify({'error': 'No hay campos para actualizar'}), 400

        params.append(user_id)

        query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            conn.commit()

            if cursor.rowcount == 0:
                return jsonify({'error': 'Usuario no encontrado'}), 404

        return jsonify({'message': 'Usuario actualizado exitosamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)