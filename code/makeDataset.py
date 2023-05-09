"""
Creation dataset from videos

"""

import cv2
from scipy import misc
from PIL import Image
import numpy as np    

def getFrames(file, destination, c):
    TAMANO_IMG=100
    print("Reading " + file)
    vidcap = cv2.VideoCapture(file)
    success,image = vidcap.read()
        
    print("Saving frames in " + destination)
    while success:
        image = cv2.resize(image, (150,100))
        cv2.imwrite(destination +  "/sign_" + sign + "_" + str(c) + ".jpg", image)      
        success,image = vidcap.read()
        c += 1
    
    print("Total count so far: " + str(c) + "\n")
    return c
    

if __name__ == '__main__':
    
    countSigns = [0, 0, 0, 0, 0]
    listSigns = ["1", "2", "3", "4", "5"]
    listOpcAC = ["_DA_", "_DB_", "_FA_", "_FB_"]
    listOpcB = ["_DA_b", "_FA_b"]

    for sign, (i, c) in zip(listSigns, enumerate(countSigns)):
        
        # Llegir opcions a + c
        for opcAC in listOpcAC:
            fileA = "dataset/videos/" + sign + opcAC + "a" + ".mp4"
            fileC = "dataset/videos/" + sign + opcAC + "c" + ".MOV"
            destination = "dataset/sign_" + sign
            c = getFrames(fileA, destination, c)
            c = getFrames(fileC, destination, c)
            
        
        # Llegir opcions b
        for opc in listOpcB:
            file = "dataset/videos/" + sign + opc + ".mp4"
            destination = "dataset/sign_" + sign
            c = getFrames(file, destination, c)
            
        countSigns[i] += c


            
    
