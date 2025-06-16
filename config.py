import os

DB_HOST = os.getenv('MYSQLHOST')
DB_USER = os.getenv('MYSQLUSER')
DB_PASSWORD = os.getenv('MYSQLPASSWORD')
DB_NAME = os.getenv('MYSQLDATABASE')
DB_PORT = int(os.getenv('MYSQLPORT',  3306))

# Rutas de carpetas
UPLOAD_FOLDER = 'imagenes'
MODEL_FOLDER = 'modelos'

# Parámetros del modelo CNN
IMG_HEIGHT = 128
IMG_WIDTH = 128