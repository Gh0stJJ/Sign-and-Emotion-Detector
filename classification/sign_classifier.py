# classification/sign_classifier.py
import numpy as np
# from tensorflow.keras.models import load_model  # Dependiendo de tu implementación
from Modelos.signClassifier import signClassifier as SignModel  # Suponiendo tu signClassifier original
from config import CLASES_GESTOS
import copy
import itertools

class SignClassifier:
    def __init__(self):
        # Carga de tu modelo (o clase) de gestos
        self.model = SignModel()

    def preprocess_landmarks(self, landmark_list):
        """
        Ajusta la lista de landmarks para que sea apta para la red:
          - Anclaje en la base.
          - Normalización.
          - Flatten.
        """
  # Realiza una copia profunda de la lista de landmarks
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Inicializa las coordenadas base
        base_x, base_y = 0, 0
        
        # Itera a través de la lista de landmarks
        for index, landmark_point in enumerate(temp_landmark_list):
            # Si es el primer landmark, establece las coordenadas base
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            # Normaliza las coordenadas restando las coordenadas base
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Aplana la lista de landmarks
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Calcula el valor máximo absoluto en la lista de landmarks
        max_value = max(list(map(abs, temp_landmark_list)))

        # Define una función para normalizar los valores
        def normalize_(n):
            return n / max_value

        # Normaliza cada valor en la lista de landmarks
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        
        # Devuelve la lista de landmarks normalizada
        return temp_landmark_list

    def predict_sign(self, processed_landmarks):
        """
        Toma la lista procesada (normalizada) y retorna (index_clase, probabilidad).
        """
        index, precision = self.model(processed_landmarks)
        return index, precision

    def get_class_label(self, index):
        return CLASES_GESTOS[index]
