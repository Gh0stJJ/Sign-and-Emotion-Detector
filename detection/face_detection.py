# detection/face_detection.py
import cv2 as cv
import numpy as np
from config import PATH_FACE_DEPLOY, PATH_FACE_WEIGHTS

class FaceDetector:
    def __init__(self, confidence_threshold=0.5):
        # Carga del modelo SSD
        self.faceNet = cv.dnn.readNet(PATH_FACE_DEPLOY, PATH_FACE_WEIGHTS)
        self.conf_threshold = confidence_threshold

    def detect_faces(self, frame):
        """
        Devuelve una lista de bounding boxes (x_inicio, y_inicio, x_final, y_final)
        para cada rostro detectado en el frame.
        """
        (h, w) = frame.shape[:2]

        # Crear blob, aquí ajustas el tamaño, normalización, etc.
        blob = cv.dnn.blobFromImage(frame, 1.0, (224, 224),
                                    (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x_i, y_i, x_f, y_f) = box.astype("int")

                # Correcciones de bordes
                x_i = max(0, x_i)
                y_i = max(0, y_i)
                x_f = min(w - 1, x_f)
                y_f = min(h - 1, y_f)

                boxes.append((x_i, y_i, x_f, y_f))

        return boxes
