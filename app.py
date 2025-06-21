# app.py
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import face_recognition
import sqlite3
import os

app = Flask(__name__)

# Configuración de la base de datos
DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
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
                embeddings BLOB
            )
        ''')
        conn.commit()

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    image_data = data['imagen_facial']
    
    # Procesar imagen y extraer embeddings
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    img_array = np.array(img)
    face_encodings = face_recognition.face_encodings(img_array)
    
    if not face_encodings:
        return jsonify({'error': 'No se detectó un rostro en la imagen'}), 400
    
    embeddings = face_encodings[0].tobytes()
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users 
                (nombre, apellido, codigo_unico, email, requisitoriado, imagen_facial, embeddings)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['nombre'],
                data['apellido'],
                data['codigo_unico'],
                data['email'],
                data['requisitoriado'],
                image_data,
                embeddings
            ))
            conn.commit()
        return jsonify({'message': 'Usuario registrado exitosamente'}), 201
    except sqlite3.IntegrityError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recognize', methods=['POST'])
def recognize_face():
    image_data = request.json['imagen']
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    img_array = np.array(img)
    face_encodings = face_recognition.face_encodings(img_array)
    
    if not face_encodings:
        return jsonify({'error': 'No se detectó un rostro en la imagen'}), 400
    
    query_embedding = face_encodings[0]
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, nombre, apellido, codigo_unico, email, requisitoriado, embeddings FROM users')
        users = cursor.fetchall()
    
    matches = []
    for user in users:
        stored_embedding = np.frombuffer(user[6], dtype=np.float64)
        distance = face_recognition.face_distance([stored_embedding], query_embedding)[0]
        
        # Umbral de coincidencia (ajustable)
        if distance < 0.6:
            matches.append({
                'id': user[0],
                'nombre': user[1],
                'apellido': user[2],
                'codigo_unico': user[3],
                'email': user[4],
                'requisitoriado': bool(user[5]),
                'confidence': 1 - distance
            })
    
    if matches:
        # Ordenar por mayor confianza
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        best_match = matches[0]
        
        if best_match['requisitoriado']:
            # Simular notificación a autoridades
            print("¡ALERTA DE SEGURIDAD! Usuario Requisitoriado Detectado. Notificación Enviada a la Policía")
        
        return jsonify({'match': True, 'user': best_match, 'all_matches': matches})
    else:
        return jsonify({'match': False})

@app.route('/users', methods=['GET'])
def get_users():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, nombre, apellido, codigo_unico, email, requisitoriado FROM users')
        users = cursor.fetchall()
    
    users_list = [{
        'id': user[0],
        'nombre': user[1],
        'apellido': user[2],
        'codigo_unico': user[3],
        'email': user[4],
        'requisitoriado': bool(user[5])
    } for user in users]
    
    return jsonify(users_list)

@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
    
    return jsonify({'message': 'Usuario eliminado exitosamente'})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)