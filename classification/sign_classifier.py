# classification/sign_classifier.py
import numpy as np
# from tensorflow.keras.models import load_model  # Dependiendo de tu implementación
from Modelos.signClassifier import signClassifier as SignModel  # Suponiendo tu signClassifier original
from config import CLASES_GESTOS
import copy
import itertools
import tensorflow as tf

class SignClassifier:
    def __init__(self, path_modelo='Modelos/Sign_classifier_V2.tflite', num_threads=1):

        # Crear un intérprete TensorFlow Lite con el modelo especificado
        self.interprete = tf.lite.Interpreter(model_path=path_modelo, num_threads=num_threads)
        # Asignar memoria para el intérprete
        self.interprete.allocate_tensors()
        # modelo
        
        # Obtener detalles de entrada y salida del intérprete
        self.entrada_detalles = self.interprete.get_input_details()
        self.salida_detalles = self.interprete.get_output_details()

    # El __call__ es un método especial que permite llamar a la instancia como si fuera una función
    def __call__(self, landmark):
        # Método para realizar inferencias con el modelo
        # Obtener el índice del tensor de entrada
        input_details_tensor_index = self.entrada_detalles[0]['index']
        # Establecer el tensor en el intérprete con los datos de "landmark"
        self.interprete.set_tensor(input_details_tensor_index, np.array([landmark], dtype=np.float32))
        # Realizar la clasificacion
        self.interprete.invoke()
        # Obtener el resultado del tensor de salida
        output_details_tensor_index = self.salida_detalles[0]['index']
        result = self.interprete.get_tensor(output_details_tensor_index)
        # Procesar el resultado y devolver el índice del resultado más alto
        result_index = np.argmax(np.squeeze(result))
        return (result_index,result[0][result_index])



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
        index, precision = self(processed_landmarks)
        return index, precision

    def get_class_label(self, index):
        return CLASES_GESTOS[index]
