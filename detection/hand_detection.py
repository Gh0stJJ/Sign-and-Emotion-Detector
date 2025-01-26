# detection/hand_detection.py
import mediapipe as mp
import cv2 as cv

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_hands(self, frame):
        """
        Recibe un frame BGR y devuelve el 'results' de MediaPipe
        que contiene los landmarks de las manos detectadas.
        """
        # MediaPipe espera un frame en RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        return results
