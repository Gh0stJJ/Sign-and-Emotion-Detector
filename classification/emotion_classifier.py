# classification/emotion_classifier.py
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from config import PATH_MODEL_EMOCIONES, CLASES_EMOCIONES

class EmotionClassifier:
    def __init__(self):
        # Carga el modelo entrenado de emociones
        self.model = load_model(PATH_MODEL_EMOCIONES)

    def predict_emotions(self, frame, face_boxes):
        """
        Dado un frame BGR y las coordenadas (xi, yi, xf, yf) de cada cara,
        retorna una lista de predicciones y ubicaciones.
        """
        preds = []
        for (xi, yi, xf, yf) in face_boxes:
            face = frame[yi:yf, xi:xf]
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            face = cv.resize(face, (48, 48))
            face_array = img_to_array(face)
            face_array = np.expand_dims(face_array, axis=0)

            # Predicción
            pred = self.model.predict(face_array, verbose=0)
            preds.append(pred[0])
        return preds

    def get_emotion_label(self, pred):
        """
        Dado un vector de predicción (por ejemplo [0.1, 0.0, 0.0, 0.7, 0.1, ...]),
        retorna la etiqueta de emoción y la probabilidad.
        """
        max_idx = np.argmax(pred)
        emotion_label = CLASES_EMOCIONES[max_idx]
        confidence = pred[max_idx]
        return emotion_label, confidence
