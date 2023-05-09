"""
Hand Clasification from segmented images

"""

import cv2
import numpy as np
import os
import random
import time
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


def extractFeatures(train, test):
    pass


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
            t = cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm)
            train.append(cv2.resize(t, (100,150)).flatten())
            
        for frame in sample[nTrain:]:
            t = cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm)
            test.append(cv2.resize(t, (100,150)).flatten())
            
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
    
    listSignsDirectory = ["signSobel_1", "signSobel_2", "signSobel_3", "signSobel_4", "signSobel_5"]
    train, trainLabels, test, testLabels = getTrainTest(listSignsDirectory, 4, 0.8,cv2.IMREAD_GRAYSCALE)
    #train, test = extractFeatures(train, test)
    
    # CLASIFICATION #####################################################################
    
    t0 = time.time()
    print("-- Model KNN --")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train, trainLabels)
    knn_predictions = knn.predict(test)
    tKnn = time.time() - t0
    
    cmKnn = confusion_matrix(testLabels, knn_predictions)
    plt.figure(figsize = (10,7))
    sn.heatmap(cmKnn, annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Knn")
    
    accKnn = knn.score(test, testLabels)
    print("Mean accuracy: " + str(accKnn))
    print("Temps: " + str(tKnn) + "\n")

    #####################################################################################

    t0 = time.time()
    print("-- Logistic Regression --")
    lr = LogisticRegression(random_state=0)
    lr.fit(train, trainLabels)
    lr_predictions = lr.predict(test)
    tLr = time.time() - t0
    
    cmLr = confusion_matrix(testLabels, lr_predictions)
    plt.figure(figsize = (10,7))
    sn.heatmap(cmLr, annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Logistic Regression")
    
    accLr = lr.score(test, testLabels)
    print("Mean accuracy: " + str(accLr))
    print("Temps: " + str(tLr) + "\n")

    #####################################################################################
    
    t0 = time.time()
    print("-- Model Gaussian Naive Bayes --")
    gnb = GaussianNB()
    gnb.fit(train, trainLabels)
    gnb_predictions = gnb.predict(test)
    tGnb = time.time() - t0
     
    cmGnb = confusion_matrix(testLabels, gnb_predictions)
    plt.figure(figsize = (10,7))
    sn.heatmap(cmGnb, annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Gaussian Naive Bayes")
    
    accGnb = gnb.score(test, testLabels)
    print("Mean accuracy: " + str(accGnb))
    print("Temps: " + str(tGnb) + "\n")
    
    #####################################################################################
    
    t0 = time.time()
    print("-- Model Decision Tree --")
    dtree = DecisionTreeClassifier(max_depth = 2)
    dtree.fit(train, trainLabels)
    dtree_predictions = dtree.predict(test) 
    tDt = time.time() - t0
    
    cmDt = confusion_matrix(testLabels, dtree_predictions)
    plt.figure(figsize = (10,7))
    sn.heatmap(cmDt, annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Decision Tree")
    
    accDt = dtree.score(test, testLabels)
    print("Mean accuracy: " + str(accDt))
    print("Temps: " + str(tDt) + "\n")
    
    #####################################################################################
    
    t0 = time.time()
    print("-- Model Random Forest Classifier --")
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(train, trainLabels)
    rf_predictions = rf.predict(test) 
    tRf = time.time() - t0
    
    cmRf = confusion_matrix(testLabels, rf_predictions)
    plt.figure(figsize = (10,7))
    sn.heatmap(cmRf, annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Random Forest Classifier")
    
    accRf = rf.score(test, testLabels)
    print("Mean accuracy: " + str(accRf))
    print("Temps: " + str(tRf) + "\n")
    
    #####################################################################################
    
    t0 = time.time()
    print("-- Model Super Vector Machine (lineal) --")
    svm = SVC(kernel = 'linear', C = 1)
    svm.fit(train, trainLabels)
    svm_predictions = svm.predict(test)
    tSvm = time.time() - t0
    
    cmSvm = confusion_matrix(testLabels, svm_predictions)
    plt.figure(figsize = (10,7))
    sn.heatmap(cmSvm, annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Super Vector Machine (lineal)")
    
    accSvm = svm.score(test, testLabels)
    print("Mean accuracy: " + str(accSvm))
    print("Temps: " + str(tSvm) + "\n")
    
    #####################################################################################

    t0 = time.time()
    print("-- Model CNN --")
    tSvm = time.time() - t0
    

    
    accSvm = svm.score(test, testLabels)
    print("Mean accuracy: " + str(accSvm))
    print("Temps: " + str(tSvm) + "\n")
    
    #####################################################################################