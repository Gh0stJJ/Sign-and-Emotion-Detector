# main.py
import cv2 as cv
import numpy as np
import tensorflow as tf
import time

# Importamos nuestros módulos personalizados
from detection.face_detection import FaceDetector
from detection.hand_detection import HandDetector
from classification.sign_classifier import SignClassifier
from classification.emotion_classifier import EmotionClassifier

# Funciones de dibujo (dibujar bounding boxes, texto, etc.)
from utils.drawing_utils import (
    draw_face_box,
    draw_text,
    draw_hand_landmarks,
    draw_hand_info
)

# Si tienes otras configuraciones globales (rutas, clases, etc.)
from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)


# Constantes de estabilidad
STABILITY_THRESHOLD = 0.75  # Tiempo minimo para confirmar un gesto
REPEATED_FRAMES = 0.75  # Tiempo minimo para repetir un gesto


def main():
    """
    Punto de entrada principal del proyecto.
    """
    #Buffer de texto para la palabra

    #Inicializamos el buffer de texto
    buffer = ""
    last_gesture_label = None

    #Iniciamos contadores de estabilidad
    current_gesture = None
    gesture_start_time = 0.0
    last_added_time = 0.0

    # 1) Configurar (opcional) uso de GPU para TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU habilitada correctamente.")
        except RuntimeError as e:
            print("Error al configurar la GPU:", e)
    else:
        print("No se detectó GPU compatible, usando CPU.")

    # 2) Inicializar detectores y clasificadores
    face_detector = FaceDetector(confidence_threshold=0.5)  # Para SSD de rostros
    hand_detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    sign_classifier = SignClassifier()          # Para clasificar señas
    emotion_classifier = EmotionClassifier()    # Para clasificar emociones

    # 3) Iniciar la captura de video
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        # 4) Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("No se obtuvo frame de la cámara.")
            break

        # 5) Voltear la imagen 
        frame = cv.flip(frame, 1)
        debug_image = frame.copy()

        # 6) DETECCIÓN DE ROSTROS
        face_boxes = face_detector.detect_faces(frame)

        # 7) CLASIFICACIÓN DE EMOCIONES para cada rostro
        face_preds = emotion_classifier.predict_emotions(frame, face_boxes)

        # 8) Dibujar resultados de rostros y emociones
        for box, pred in zip(face_boxes, face_preds):
            emotion_label, confidence = emotion_classifier.get_emotion_label(pred)
            label_text = f"{emotion_label}: {confidence*100:.0f}%"
            draw_face_box(debug_image, box, label=label_text, color=(0, 255, 0))

        # 9) DETECCIÓN DE MANOS (MediaPipe)
        results = hand_detector.detect_hands(frame)

        # Procesar cada mano detectada
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # a) Dibujar landmarks de la mano (opcional)
                draw_hand_landmarks(debug_image, hand_landmarks)

                # b) Convertir landmarks a lista normalizada, etc.
                #    Estas funciones puedes definirlas en sign_classifier
                #    o en tu hand_detection. Ejemplo:
                landmark_list = hand_detector.get_landmark_list(debug_image, hand_landmarks)
                processed_lms = sign_classifier.preprocess_landmarks(landmark_list)

                # c) Clasificar la seña
                index, precision = sign_classifier.predict_sign(processed_lms)
                gesture_label = sign_classifier.get_class_label(index)

    
                #Estabilidad del gesto

                if gesture_label != current_gesture:
                    #Reiniciamos el contador de tiempo
                    current_gesture = gesture_label
                    gesture_start_time = time.time()
                    last_added_time = 0.0 # Reseteamos el tiempo de la ultima letra

                #Calcular cuanto ha estado el gesto estable
                gesture_stability_time = time.time() - gesture_start_time


                #Checar el tiempo de estabilidad
                if gesture_stability_time > STABILITY_THRESHOLD:
                    now = time.time()

                    if (last_added_time == 0.0) or (now - last_added_time >= REPEATED_FRAMES):
                        
                        #Escribir la palabra en pantalla
                        
                        if gesture_label == "space":
                            buffer += " "
                        elif gesture_label == "delete":
                            if len(buffer) > 0:
                                buffer = buffer[:-1]
                        else:
                            buffer += gesture_label

                        #Actualizamos el tiempo de la ultima letra
                        last_added_time = now

                        #Guardamos el ultimo gesto para evitar repeticiones
                        last_gesture_label = gesture_label

                    
                # d) Dibujar info de la mano (mano derecha/izquierda, letra, etc.)
                mano_text = handedness.classification[0].label  # 'Left' o 'Right'
                draw_hand_info(debug_image, hand_landmarks, mano_text, gesture_label)

                # e) Mostrar la letra y su precisión en alguna parte de la pantalla
                draw_text(debug_image, f"Letra: {gesture_label} ({precision:.2f})", pos=(10, 100))

        #  Mostrar buffer de texto
        cv.rectangle(debug_image, (10, 10), (400, 60), (120,120,120), -1)

        # (B) ESCRIBIR TEXTO ENCIMA DEL FONDO
        cv.putText(debug_image, buffer, (20, 45),
            cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

        # 10) Mensaje de salir
        draw_text(debug_image, "Salir (Escape Key)", pos=(5, 20))

        # 11) Mostrar resultado
        cv.imshow("Sign and Emotion Detector 2.0", debug_image)

        # 12) Salir con tecla ESC
        key = cv.waitKey(10)
        if key == 27:  # 27 = ESC
            break

    # Liberar recursos
    cap.release()
    cv.destroyAllWindows()

# Para ejecutar el script como programa principal:
if __name__ == "__main__":
    main()
