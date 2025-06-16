# Usa imagen oficial de Python 3.10
FROM python:3.10-slim

# Instala dependencias de sistema necesarias (incluye libGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia todo el contenido de tu proyecto al contenedor
COPY . .

# Instala dependencias Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto (lo usará Railway)
EXPOSE $PORT

# Comando para correr Flask con Gunicorn en Railway
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]