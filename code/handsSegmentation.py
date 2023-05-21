"""
Hand Segmentation

"""

import os
import cv2
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)


def Sobel(src):
    """
    Esta función se encarga de aplicar el algoritmo de Sobel

    Argumentos:
    - src: Imagen a segmentar
        
    Retorna:
    - grad: Imagen resultante

    """
    # Definir elemento estructurante
    kernel = np.ones((30,30), np.uint8)
    kernel2 = np.ones((2,2), np.uint8)
    
    src = cv2.GaussianBlur(src, (3, 3), 0)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    window_name = ('Sobel Demo - Simple Edge Detector')  
        
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Umbralitzacio de la imagen
    _, grad = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
    grad = cv2.erode(grad, kernel2, iterations=2)
    
    return grad


def color_filter(src):
    """
    Esta función se encarga de aplicar la segmentacion por color

    Argumentos:
    - src: Imagen a segmentar
        
    Retorna:
    - hand: Imagen resultante

    """
    img_HSV = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)

    Lfilter = (0,10,20)
    Hfilter = (20,155,255)

    mask = cv2.inRange(img_HSV,Lfilter,Hfilter)
    hand = cv2.bitwise_and(src, src, mask=mask)

    kernel = np.ones((3, 3), np.uint8)
    hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, kernel)
    hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, kernel)

    return hand


def canny(src):
    """
    Esta función se encarga de aplicar el algoritmo de Canny

    Argumentos:
    - src: Imagen a segmentar
        
    Retorna:
    - cannyIm: Imagen resultante

    """
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    gauss = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cannyIm = cv2.Canny(gauss, 50, 150)

    return cannyIm


def otsu(src):
    """
    Esta función se encarga de aplicar el algoritmo de Otsu

    Argumentos:
    - src: Imagen a segmentar
        
    Retorna:
    - hand: Imagen resultante

    """
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hand = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations=2)

    return hand



if __name__ == '__main__':
    
    listSignsDirectory = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5"]
    listSignsDirectorySobel = ["signSobel_1", "signSobel_2", "signSobel_3", "signSobel_4", "signSobel_5"]
    listSignsDirectoryColorFilter = ["signCF_1", "signCF_2", "signCF_3", "signCF_4", "signCF_5"]
    listSignsDirectoryCanny = ["signCanny_1", "signCanny_2", "signCanny_3", "signCanny_4", "signCanny_5"]
    listSignsDirectoryOtsu = ["signOtsu_1", "signOtsu_2", "signOtsu_3", "signOtsu_4", "signOtsu_5"]


    print("Applying Sobel to frames...\n")
    for directory, new in zip(listSignsDirectory, listSignsDirectorySobel):
        
        relativePath = "../dataset/" + directory
        print("Transforming images from " + relativePath)
        
        for frame in range(len(os.listdir(os.path.abspath(relativePath)))): 
            imSobel = Sobel(cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg"))
            
            cv2.imwrite("../dataset/" + new + "/" + new + "_" + str(frame) + ".jpg", imSobel)
        
        print("New images saved in " + new + "\n")
    
    
    print("Applying Color Filter to frames...\n")
    for directory, new in zip(listSignsDirectory, listSignsDirectoryColorFilter):

        relativePath = "../dataset/" + directory
        print("Transforming images from " + relativePath)

        for frame in range(len(os.listdir(os.path.abspath(relativePath)))):
            imCF = color_filter(cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg"))

            cv2.imwrite("../dataset/" + new + "/" + new + "_" + str(frame) + ".jpg", imCF)

        print("New images saved in " + new + "\n")

    
    print("Applying Canny to frames...\n")
    for directory, new in zip(listSignsDirectory, listSignsDirectoryCanny):

        relativePath = "../dataset/" + directory
        print("Transforming images from " + relativePath)

        for frame in range(len(os.listdir(os.path.abspath(relativePath)))):
            imCanny = canny(cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg"))

            cv2.imwrite("../dataset/" + new + "/" + new + "_" + str(frame) + ".jpg", imCanny)

        print("New images saved in " + new + "\n")


    print("Applying Otsu to frames...\n")
    for directory, new in zip(listSignsDirectory, listSignsDirectoryOtsu):

        relativePath = "../dataset/" + directory
        print("Transforming images from " + relativePath)

        for frame in range(len(os.listdir(os.path.abspath(relativePath)))):
            imOtsu = otsu(cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg"))

            cv2.imwrite("../dataset/" + new + "/" + new + "_" + str(frame) + ".jpg", imOtsu)

        print("New images saved in " + new + "\n")


        
        