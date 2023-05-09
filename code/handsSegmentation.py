"""
Hand Segmentation

"""

import os
import cv2
import numpy as np


def Sobel(src):
    
    # definir elemento estructurante
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


if __name__ == '__main__':
    
    listSignsDirectory = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5"]
    listSignsDirectorySobel = ["signSobel_1", "signSobel_2", "signSobel_3", "signSobel_4", "signSobel_5"]
    
    print("Applying Sobel to frames...\n")
    for directory, new in zip(listSignsDirectory, listSignsDirectorySobel): #Nueva lista en zip
        
        relativePath = "../dataset/" + directory
        print("Transforming images from " + relativePath)
        
        for frame in range(len(os.listdir(os.path.abspath(relativePath)))): 
            imSobel = Sobel(cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg"))
            #imNueva = ...
            
            cv2.imwrite("../dataset/" + new + "/" + new + "_" + str(frame) + ".jpg", imSobel)
            #cv2.imwrite ...
        
        print("New images saved in " + new + "\n")
        
        