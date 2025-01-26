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
