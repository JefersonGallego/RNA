import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# Cargar las imágenes y crear las etiquetas
images = []
labels = []
for i in range(10):
    img = cv2.imread(f'images/image_{i}.png', cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0  # Normalizar las imágenes
    images.append(img)
    labels.append(i)

images = np.array(images).reshape(-1, 20, 20, 1) # aplanar vector de imagenes
labels = np.array(labels)

# Crear el modelo de red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(20, 20, 1)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(images, labels, epochs=1000)

# Guardar el modelo en el formato nativo de Keras
model.save('simple_model.keras')

# Crear archivo de etiquetas
with open('labels.txt', 'w') as f:
    for i in range(10):
        f.write(f"Class {i}\n")
