# utils/drawing_utils.py
import cv2 as cv

def draw_face_box(image, box, label="", color=(255,0,0)):
    (xi, yi, xf, yf) = box
    # Dibujamos el rectángulo
    cv.rectangle(image, (xi, yi), (xf, yf), color, 2)
    if label:
        cv.rectangle(image, (xi, yi-30), (xf, yi), color, -1)
        cv.putText(image, label, (xi+5, yi-5), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255,255,255), 1, cv.LINE_AA)

def draw_text(image, text, pos=(5,20), color=(255,255,255), bg_color=(0,0,0)):
    cv.rectangle(image, (pos[0], pos[1]-15), (pos[0]+200, pos[1]+5), bg_color, -1)
    cv.putText(image, text, pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

def draw_hand_landmarks(image, hand_landmarks, color=(0, 255, 255), radius=5, thickness=2):
    """
    Dibuja los landmarks y las conexiones de una mano en la imagen.
    - image: frame de OpenCV en BGR.
    - hand_landmarks: objeto 'hand_landmarks' que retorna MediaPipe.
    - color: color de las líneas y puntos (B, G, R).
    - radius: radio de los puntos.
    - thickness: grosor de las líneas.
    """
    # Obtiene la lista de conexiones y landmarks de MediaPipe

    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    return image

def draw_hand_info(image, hand_landmarks, mano_label, gesture_label):
    """
    Dibuja en la imagen la información de la mano (ej. 'Left/Right' y la letra reconocida).
    - image: el frame BGR de OpenCV
    - hand_landmarks: el objeto que retorna MediaPipe con los landmarks de la mano
    - mano_label: texto a mostrar (ej: 'Left' o 'Right')
    - gesture_label: la letra o gesto que se reconoció
    """
    h, w = image.shape[:2]

    # Tomemos como referencia el landmark de la muñeca (wrist) para ubicar el texto
    wrist = hand_landmarks.landmark[0]
    x_text = int(wrist.x * w)
    y_text = int(wrist.y * h)

    info_text = f"{mano_label} - {gesture_label}"
    cv.putText(image, info_text, (x_text, y_text),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)