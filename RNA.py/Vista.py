import cv2
import numpy as np
import tensorflow as tf

class Classifier:
    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        np.set_printoptions(suppress=True)
        self.model = tf.keras.models.load_model(self.model_path)
        self.data = np.ndarray(shape=(1, 20, 20, 1), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            with open(self.labels_path, "r") as label_file:
                self.list_labels = [line.strip() for line in label_file]
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        imgS = cv2.resize(img, (20, 20))
        image_array = np.asarray(imgS).astype(np.float32) / 255.0
        self.data[0] = np.expand_dims(image_array, axis=-1)
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)
        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]), pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)
        return list(prediction[0]), indexVal
# Cargar el modelo y las etiquetas

classifier = Classifier('simple_model.keras', 'labels.txt')

# Inicializar la captura de video
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara predeterminada

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar y convertir a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (20, 20))

    # Hacer la predicción
    prediction, index = classifier.getPrediction(resized_frame, draw=False)

    # Mostrar la predicción en el fotograma
    label = classifier.list_labels[index]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
