# Video Blur Editor (YOLOv8 + OpenCV)

Aplicación en Python para detectar formas, rostros y aplicar difuminado manual o automático en videos utilizando YOLOv8 y OpenCV.

## Features

- detección de rostros con YOLO
- procesamiento de video
- exportación en múltiples resoluciones
- soporte GPU AMD (DIRECTML)
- interfaz gráfica

## Requisitos del sistema

- Python 3.10.11
- FFmpeg instalado

Instalación en Windows:

1. Descargar FFmpeg
https://ffmpeg.org/download.html

2. Agregar ffmpeg al PATH

## Instalación

git clone https://github.com/luisdl-dev/videoeditor-difuminador

cd video-blur-editor

pip install -r requirements.txt

⚠️ Después de instalar, desinstala conflictos relacionados a otras versiones de opencv-python si se instaló automáticamente. Esto evita conflictos con opencv-contrib-python y problemas de trackers/difuminados:
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python -m pip uninstall -y numpy

y vuelve a instalar:
python -m pip install numpy==1.26.4
python -m pip install opencv-contrib-python==4.7.0.72

Modo GPU (AMD / DirectML, solo si tiene tarjeta dedicada AMD):
pip install torch-directml

## Uso

python src/main.py

## Demo
(en proceso)


