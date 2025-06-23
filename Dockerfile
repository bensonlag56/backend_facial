FROM python:3.12-slim

# Instala dependencias necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Instala tus dependencias Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto para Railway
ENV PORT 5000
EXPOSE 5000

CMD ["python", "app.py"]