"""
Hand Clasification from segmented images

"""

import cv2
import numpy as np
import os
import random

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def getTrainTest(listSignsDirectory, sampleDiv, pTrain, typeIm):
    
    train = []
    test = []
    trainLabels = []
    testLabels = []
    
    for directory in listSignsDirectory:
        
        relativePath = "../dataset/" + directory
        
        # Get number frames
        nFiles = len(os.listdir(os.path.abspath(relativePath)))   
        
        # Random sample of Data for Train - Test
        sample = random.sample(range(1,nFiles), round(nFiles/sampleDiv))
        nTrain = int(round(round(nFiles/sampleDiv)*pTrain))
        
        print("Reading " + str(round(nFiles/sampleDiv)) + " random frames from " + relativePath)
        
        # Reading random sample
        for frame in sample[:nTrain]:
            train.append(cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm))
            
        for frame in sample[nTrain:]:
            test.append(cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm))
            
        trainLabels.append( [directory] * nTrain)
        testLabels.append( [directory] * (round(nFiles/sampleDiv) - nTrain))
        
    
    trainLabels = [item for sublist in trainLabels for item in sublist]
    testLabels = [item for sublist in testLabels for item in sublist]
    
    print("Shuffling data...\n")   
    zipped = list(zip(train, trainLabels))
    random.shuffle(zipped)
    train, trainLabels = zip(*zipped)

    zipped = list(zip(test, testLabels))
    random.shuffle(zipped)
    test, testLabels = zip(*zipped)
    
    return train, trainLabels, test, testLabels



if __name__ == '__main__':
    
    listSignsDirectory = ["signBin_1", "signBin_2", "signBin_3", "signBin_4", "signBin_5"]
    train, trainLabels, test, testLabels = getTrainTest(listSignsDirectory, 4, 0.8,cv2.IMREAD_GRAYSCALE)
    
    
    # CLASIFICATION #####################################################################################
    
    print("-- Model KNN --")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train, trainLabels)
    knn_predictions = knn.predict(test)
    cmKnn = confusion_matrix(testLabels, knn_predictions)
    accKnn = knn.score(test, testLabels)
    print("Mean accuracy: " + str(accKnn) + "\n")

    #print("-- Logistic Regression --")
    #lr = LogisticRegression(random_state=0)
    #lr.fit(train, trainLabels)
    #lr_predictions = lr.predict(test)
    #cmLr = confusion_matrix(testLabels, lr_predictions)
    #accLr = lr.score(test, testLabels)
    #print("Mean accuracy: " + str(accLr) + "\n")
    
    #print("-- Model Gaussian Naive Bayes --")
    #gnb = GaussianNB()
    #gnb.fit(train, trainLabels)
    #gnb_predictions = gnb.predict(test)
    #cmGnb = confusion_matrix(testLabels, gnb_predictions)
    #accGnb = gnb.score(test, testLabels)
    #print("Mean accuracy: " + str(accGnb) + "\n")
    
    #print("-- Model Decision Tree --")
    #dtree = DecisionTreeClassifier(max_depth = 2)
    #dtree.fit(train, trainLabels)
    #dtree_predictions = dtree.predict(test)  
    #cmDt = confusion_matrix(testLabels, dtree_predictions)
    #accDt = dtree.score(test, testLabels)
    #print("Mean accuracy: " + str(accDt) + "\n")
    
    #print("-- Model Random Forest Classifier --")
    #rf = RandomForestClassifier(max_depth=2, random_state=0)
    #rf.fit(train, trainLabels)
    #rf_predictions = rf.predict(test)  
    #cmRf = confusion_matrix(testLabels, rf_predictions)
    #accRf = rf.score(test, testLabels)
    #print("Mean accuracy: " + str(accRf) + "\n")
    
    #print("-- Model Super Vector Machine (lineal) --")
    #svm = SVC(kernel = 'linear', C = 1)
    #svm.fit(train, trainLabels)
    #svm_predictions = svm.predict(test)
    #cmSvm = confusion_matrix(testLabels, svm_predictions)
    #accSvm = svm.score(test, testLabels)
    #print("Mean accuracy: " + str(accSvm) + "\n")
    
    
