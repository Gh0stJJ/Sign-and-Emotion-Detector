import numpy as np
import tensorflow as tf

class signClassifier(object):

    def __init__(self, path_modelo='Modelos/Sign_classifier_V2.tflite', num_threads=1):

        # Crear un intérprete TensorFlow Lite con el modelo especificado
        self.interprete = tf.lite.Interpreter(model_path=path_modelo, num_threads=num_threads)
        # Asignar memoria para el intérprete
        self.interprete.allocate_tensors()
        # Obtener detalles de entrada y salida del intérprete
        self.entrada_detalles = self.interprete.get_input_details()
        self.salida_detalles = self.interprete.get_output_details()

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