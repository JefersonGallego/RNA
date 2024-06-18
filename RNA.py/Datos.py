import numpy as np
import cv2
import os

# Crear una carpeta para almacenar las imágenes
os.makedirs('images', exist_ok=True)

# Generar 10 imágenes de 5x5 píxeles en escala de grises
for i in range(10):
    img = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    cv2.imwrite(f'images/image_{i}.png', img)
