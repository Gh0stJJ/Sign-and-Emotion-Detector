# config.py

# Rutas de modelos
PATH_MODEL_EMOCIONES = "Modelos/modelo_emociones06.h5"
PATH_FACE_DEPLOY = "Modelos/deploy.prototxt"
PATH_FACE_WEIGHTS = "Modelos/res10_300x300_ssd_iter_140000.caffemodel"
SIGN_PRO_MODEL = 'Modelos/Sign_classifier_V2.1.tflite'

# Clases para la detección de manos y emociones
CLASES_GESTOS = [
    'space','delete','a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z'
]
CLASES_EMOCIONES = [
    'enojo 😡','disgusto 😐','miedo 😱','felicidad 😁','neutro 🤨','tristeza 😥','sorpresa 😧'
]

# Otras configuraciones
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
