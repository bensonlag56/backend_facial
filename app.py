from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from utils.cnn_model import train_and_save_model
from utils.database import insert_usuario, delete_usuario, get_all_usuarios
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'imagenes'
MODEL_FOLDER = 'modelos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/registrar_usuario', methods=['POST'])
def registrar_usuario():
    nombre = request.form.get('nombre')
    apellido = request.form.get('apellido')
    codigo_unico = request.form.get('codigo_unico')
    email = request.form.get('email')
    requisitoriado = request.form.get('requisitoriado') == 'true'

    if 'imagenes' not in request.files:
        return jsonify({'error': 'No se enviaron imágenes'}), 400

    files = request.files.getlist('imagenes')
    if len(files) < 5:
        return jsonify({'error': 'Se requieren al menos 5 imágenes'}), 400

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], codigo_unico)
    class_folder = os.path.join(user_folder, 'rostro')  # subfolder for class
    os.makedirs(class_folder, exist_ok=True)

    for idx, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{idx}_{file.filename}")
            file_path = os.path.join(class_folder, filename)
            file.save(file_path)

    model_path = os.path.join(app.config['MODEL_FOLDER'], f"{codigo_unico}_modelo.h5")
    train_and_save_model(user_folder, model_path)

    success = insert_usuario(nombre, apellido, codigo_unico, email, requisitoriado, model_path)
    if success:
        return jsonify({'message': 'Usuario registrado y modelo creado exitosamente'})
    else:
        return jsonify({'error': 'Error al registrar en la base de datos'}), 500

@app.route('/eliminar_usuario', methods=['DELETE'])
def eliminar_usuario():
    codigo_unico = request.args.get('codigo_unico')
    if not codigo_unico:
        return jsonify({'error': 'codigo_unico es requerido'}), 400

    success = delete_usuario(codigo_unico)
    if success:
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"{codigo_unico}_modelo.h5")
        if os.path.exists(model_path):
            os.remove(model_path)
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], codigo_unico)
        if os.path.exists(user_folder):
            for file in os.listdir(user_folder):
                os.remove(os.path.join(user_folder, file))
            os.rmdir(user_folder)
        return jsonify({'message': 'Usuario y archivos eliminados correctamente'})
    else:
        return jsonify({'error': 'Usuario no encontrado o no se pudo eliminar'}), 404

# Endpoint para listar todos los usuarios registrados
@app.route('/listar_usuarios', methods=['GET'])
def listar_usuarios():
    usuarios = get_all_usuarios()
    return jsonify(usuarios)

import cv2
import numpy as np
from tensorflow.keras.models import load_model

@app.route('/reconocer_usuario', methods=['POST'])
def reconocer_usuario():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    file = request.files['imagen']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        # Preprocesar la imagen
        img = cv2.imread(temp_path)
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        usuarios = get_all_usuarios()
        for usuario in usuarios:
            model_path = usuario['ruta_modelo']
            if os.path.exists(model_path):
                model = load_model(model_path)
                prediction = model.predict(img)
                confidence = prediction[0][0]  # Probabilidad para la única clase entrenada

                # Aplicar un umbral más estricto y comprobación para reducir falsos positivos
                if 0.8 < confidence <= 1.0:  
                    os.remove(temp_path)
                    return jsonify({
                        'message': 'Usuario reconocido',
                        'codigo_unico': usuario['codigo_unico'],
                        'nombre': usuario['nombre'],
                        'apellido': usuario['apellido'],
                        'requisitoriado': usuario['requisitoriado']
                    })

        os.remove(temp_path)
        return jsonify({'message': 'Usuario no reconocido'})
    else:
        return jsonify({'error': 'Formato de imagen no permitido'}), 400

if __name__ == '__main__':
    app.run(debug=True)
