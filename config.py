import os

DB_HOST = os.getenv('mysql.railway.internal')
DB_USER = os.getenv('root')
DB_PASSWORD = os.getenv('JqsHAilfOvOfsxBJSzyKjQnWOTshsNAz')
DB_NAME = os.getenv('railway')
DB_PORT = int(os.getenv('3306', 3306))

# Rutas de carpetas
UPLOAD_FOLDER = 'imagenes'
MODEL_FOLDER = 'modelos'

# Parámetros del modelo CNN
IMG_HEIGHT = 128
IMG_WIDTH = 128