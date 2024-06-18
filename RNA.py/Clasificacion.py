import numpy as np
import tensorflow as tf
import cv2

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

# Simular la captura de una imagen de 5x5 píxeles
img = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
prediction, index = classifier.getPrediction(img, draw=False)
print(prediction)
