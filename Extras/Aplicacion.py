import copy
import csv
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
import Modelos.ClasificadorSenias as clas

#Se recogen 500 instancias por cada letra
CLASES=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u', 'v','w','x','y','z']
def main():
    cap = cv.VideoCapture(0)#Obtener la camara
    mp_hands = mp.solutions.hands #
    hands = mp_hands.Hands(
        max_num_hands=2,#Solo detecta una mano
        min_detection_confidence=0.5,#Confianza minima para detectar una mano
        min_tracking_confidence=0.5,#Confianza minima para seguir una mano
    )
    clasificador = clas.ClasificadorSenias()#Clasificador de senias
    while True:
        key = cv.waitKey(10)#Espera 10 milisegundos por tecla presionada
        if key == 27:  # tecla ESC, salir
            break
        ret, image = cap.read()#Obtener imagen de la camara
        if not ret:#Si no se obtiene imagen, salir
            break
        image = cv.flip(image, 1)  # Voltear imagen
        debug_image = copy.deepcopy(image)#Crear copia de la imagen sin procesarla
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)#Convertir imagen a RGB
        image.flags.writeable = False #Desactivar escritura de la imagen
        results = hands.process(image) #Detectar manos y obtener sus puntos clave
        image.flags.writeable = True #Activar escritura de la imagen
        if results.multi_hand_landmarks is not None:#Si se detecta una mano
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                # Calcular el borde de la mano
                brect = calculo_borde(debug_image, hand_landmarks)
                # Calculo de puntos clave de la mano
                landmark_list = lista_landmark(debug_image, hand_landmarks)
                 # Conversion de datos a coordenadas relativas y normalizar
                pre_processed_landmark_list = procesar_landmark(landmark_list)
                # Clasificar la seña de la mano
                index,precision = clasificador(pre_processed_landmark_list)
                # Graficar resultados y caracteristicas
                debug_image = dibujar_borde(True, debug_image, brect)
                debug_image = dibujar_landmarks(debug_image, landmark_list)
                debug_image = dibujar_info_borde(debug_image,brect, handedness,CLASES[index])
                mostrar_resultado(debug_image,CLASES[index],precision)       
        mostrar_info(debug_image)
        cv.imshow('Deteccion de Senias', debug_image)

    cap.release()
    cv.destroyAllWindows()

def calculo_borde(image, landmarks):
    # Obtiene el ancho y alto de la imagen
    image_w, image_h = image.shape[1], image.shape[0]

    # Crea un array vacío para almacenar las coordenadas de los landmarks
    landmark_array = np.empty((0, 2), int)
    
    # Itera a través de los landmarks y calcula las coordenadas de cada landmark en píxeles en la imagen
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_w), image_w - 1)
        landmark_y = min(int(landmark.y * image_h), image_h- 1)
        landmark_punto = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_punto, axis=0)
    
    # Utiliza la función "cv.boundingRect" para calcular el borde alrededor de los landmarks y obtener las coordenadas del rectángulo delimitador (x, y, w, h)
    x, y, w, h = cv.boundingRect(landmark_array)
    
    # Devuelve las coordenadas del borde como una lista [x, y, x + w, y + h]
    return [x, y, x + w, y + h]

def lista_landmark(image, landmarks):
    # Obtiene el ancho y alto de la imagen
    image_width, image_height = image.shape[1], image.shape[0]
    
    # Crea una lista vacía para almacenar las coordenadas de los landmarks
    landmark_point = []
    
    # Itera a través de los landmarks y calcula las coordenadas de cada landmark en píxeles en la imagen
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    # Devuelve la lista de coordenadas de los landmarks
    return landmark_point

def procesar_landmark(landmark_list):
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

def dibujar_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Pulgar
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Dedos
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Dedo del medio
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Dedo anular
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Meñique
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palma
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Puntos clave
    for index, landmark in enumerate(landmark_point):
        if index == 0: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def dibujar_borde(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0]+10, brect[1]-10), (brect[2]+10, brect[3]-10),(0, 0, 0), 1)
    return image

def mostrar_resultado(image,letra="",precision=""):
    cv.rectangle(image, (5, 30), (200, 80),(0, 0, 0), -1)
    cv.putText(image, "Letra: "+letra, (5, 50), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(image, "Precision: " + str(precision), (5, 70), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
    return image

def mostrar_info(image):
    cv.rectangle(image, (5, 0), (200, 30),(0, 0, 0), -1)
    cv.putText(image, "Salir (ESC)", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
    return image

def dibujar_info_borde(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),(0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = "Mano: "+info_text + ' Letra: ' + str.upper(hand_sign_text)
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()