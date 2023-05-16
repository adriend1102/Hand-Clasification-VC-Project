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
import tensorflow as tf
import random
import cv2

from skimage.feature import hog

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

def plot_image(i, predictions_array, true_labels, images):
    """
    Esta función se encarga de mostrar una imagen junto con su etiqueta de predicción.

    Argumentos:
    - i: Índice de la imagen dentro de las listas predictions_array, true_labels y images.
    - predictions_array: Un arreglo de predicciones para cada imagen.
    - true_labels: Un arreglo de las etiquetas reales de las imágenes.
    - images: Un arreglo de las imágenes a mostrar.

    Retorna:
    - None

    """
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    index = np.where(true_label == 1)[0][0]
    #index += 1
    #print(index)
    predicted_label = np.argmax(predictions_array)
    #print(predicted_label)
    if predicted_label == index:
        color = 'blue'
    else:
        color = 'red'
    #print(class_names[predicted_label])
    plt.xlabel(f"Prediccion: {predicted_label+1}", color=color)

def plot_value_array(i, predictions_array, true_label):
    """
    Esta función se encarga de mostrar un gráfico de barras que representa los valores de confianza de las predicciones.

    Argumentos:
    - i: Índice del elemento dentro de las listas predictions_array y true_label.
    - predictions_array: Un arreglo de predicciones para cada elemento.
    - true_label: Un arreglo de las etiquetas reales de los elementos.

    Retorna:
    - None

    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    #print(predictions_array.max())
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#888888")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    index = np.where(true_label == 1)[0][0]
    
    thisplot[predicted_label].set_color('red')
    thisplot[index].set_color('blue')

# Obtenemos las predicciones del modelo utilizando X_test como entrada

def tablaAciertos(predictions, labels, imatges):
    """
    Esta función se encarga de mostrar una tabla de imágenes con sus etiquetas de predicción y los valores de confianza.

    Argumentos:
    - predictions: Un arreglo de predicciones para cada imagen.
    - labels: Un arreglo de las etiquetas reales de las imágenes.
    - imatges: Un arreglo de las imágenes a mostrar.

    Retorna:
    - None

    """
    numrows=5
    numcols=3
    numimages = numrows*numcols

    plt.figure(figsize=(2*2*numcols, 2*numrows))

    # Iteramos sobre las primeras 15 imágenes y etiquetas de X_test y y_test
    for i in range(15):
      plt.subplot(numrows, 2*numcols, 2*i+1)
      plot_image(i, predictions, labels, imatges)
      plt.subplot(numrows, 2*numcols, 2*i+2)
      plot_value_array(i, predictions, labels)

    plt.show()
class myCallback(tf.keras.callbacks.Callback):
    """
    clase Callback para limitante
    """
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()
def getTrainTestCNN(listSignsDirectory, sampleDiv, pTrain, typeIm):
    
    train = []
    test = []
    trainLabels = []
    testLabels = []
    
    for directory in listSignsDirectory:
        
        relativePath = "dataset/" + directory
        
        # Get number frames
        nFiles = len(os.listdir(os.path.abspath(relativePath)))   
        
        # Random sample of Data for Train - Test
        sample = random.sample(range(1,nFiles), round(nFiles/sampleDiv))
        nTrain = int(round(round(nFiles/sampleDiv)*pTrain))
        
        print("Reading " + str(round(nFiles/sampleDiv)) + " random frames from " + relativePath)
        
        # Reading random sample
        for frame in sample[:nTrain]:
            img = cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train.append(img)
            
        for frame in sample[nTrain:]:
            img = cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test.append(img)
        #Hacemos one_hot encode
        one_hot = [0] * 5
        one_hot[int(directory[-1]) - 1] = 1
        trainLabels.append( [one_hot] * nTrain)
        testLabels.append( [one_hot] * (round(nFiles/sampleDiv) - nTrain))
        
    
    trainLabels = [item for sublist in trainLabels for item in sublist]
    testLabels = [item for sublist in testLabels for item in sublist]
    #Hacemos one_hot encode
    #one_hot = [0] * 5
    #one_hot[code - 1] = 1
    #y.append(one_hot)
    
    print("Shuffling data...\n")   
    zipped = list(zip(train, trainLabels))
    random.shuffle(zipped)
    train, trainLabels = zip(*zipped)

    zipped = list(zip(test, testLabels))
    random.shuffle(zipped)
    test, testLabels = zip(*zipped)
    
    return train, trainLabels, test, testLabels

# Split dataset
def getTrainTest(listSignsDirectory, sampleDiv, pTrain, typeIm):
    
    train = []
    test = []
    trainLabels = []
    testLabels = []
    
    for directory in listSignsDirectory:
        
        relativePath = "dataset/" + directory
        
        # Get number frames
        nFiles = len(os.listdir(os.path.abspath(relativePath)))   
        
        # Random sample of Data for Train - Test
        sample = random.sample(range(1,nFiles), round(nFiles/sampleDiv))
        nTrain = int(round(round(nFiles/sampleDiv)*pTrain))
        
        print("Reading " + str(round(nFiles/sampleDiv)) + " random frames from " + relativePath)
        
        # Reading random sample
        for frame in sample[:nTrain]:
            t = cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm)
            train.append(t)
            
        for frame in sample[nTrain:]:
            t = cv2.imread(relativePath + "/" + directory + "_" + str(frame) + ".jpg", typeIm)
            test.append(t)
            
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


# Get Hog descriptor
def hogDescriptor(train, test):
    
    print("Obtaining hog feature descriptors...")
    
    featuresTrain = []
    featuresTest = [] 
    
    # Train
    for frame in train:
        fd, im = hog(frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        featuresTrain.append(fd)
    
    # Test
    for frame in test:
        fd, im = hog(frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        featuresTest.append(fd)
    
    return featuresTrain, featuresTest



if __name__ == '__main__':
    
    listSignsDirectory = ["signSobel_1", "signSobel_2", "signSobel_3", "signSobel_4", "signSobel_5"]
    #listSignsDirectory = ["signCF_1", "signCF_2", "signCF_3", "signCF_4", "signCF_5"]
    #listSignsDirectory = ["signCanny_1", "signCanny_2", "signCanny_3", "signCanny_4", "signCanny_5"]
    #listSignsDirectory = ["signOtsu_1", "signOtsu_2", "signOtsu_3", "signOtsu_4", "signOtsu_5"]
    train, trainLabels, test, testLabels = getTrainTest(listSignsDirectory, 4, 0.8, cv2.IMREAD_GRAYSCALE)
    
    # Obtain features
    trainHOG, testHOG = hogDescriptor(train, test)
      
    # Flatten train - test
    train = list(map(lambda x: cv2.resize(x, (x.shape[0], x.shape[1])).flatten(), train))
    test = list(map(lambda x: cv2.resize(x, (x.shape[0], x.shape[1])).flatten(), test))
    
    
    # CLASIFICATION #####################################################################

    t0 = time.time()
    print("\n------ Model KNN ------")
    
    print("Sense Features")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train, trainLabels)
    knn_predictions = knn.predict(test)
    tKnn = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, knn_predictions), annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Knn")
    print("\n", classification_report(testLabels, knn_predictions))
    print("Temps: " + str(tKnn) + "\n")


    print("Amb HOG Features")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(trainHOG, trainLabels)
    knn_predictionsHOG = knn.predict(testHOG)
    tKnn = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, knn_predictionsHOG), annot=True,cmap="OrRd")
    plt.title("Confusion Matrix Knn + HOG Features")
    print("\n", classification_report(testLabels, knn_predictionsHOG))
    print("Temps: " + str(tKnn) + "\n")
         
    
    #####################################################################################

    t0 = time.time()
    print("\n------ Logistic Regression ------")
    
    print("Sense Features")
    lr = LogisticRegression(random_state=0)
    lr.fit(train, trainLabels)
    lr_predictions = lr.predict(test)
    tLr = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, lr_predictions), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Logistic Regression") 
    print("\n", classification_report(testLabels, lr_predictions))
    print("Temps: " + str(tLr) + "\n")
    
    
    print("Amb Features")
    lr = LogisticRegression(random_state=0)
    lr.fit(trainHOG, trainLabels)
    lr_predictionsHOG = lr.predict(testHOG)
    tLr = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, lr_predictionsHOG), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Logistic Regression + HOG Features")
    print("\n", classification_report(testLabels, lr_predictionsHOG))
    print("Temps: " + str(tLr) + "\n")

    
    #####################################################################################
    
    t0 = time.time()
    print("\n------ Model Gaussian Naive Bayes ------")
    
    print("Sense Features")
    gnb = GaussianNB()
    gnb.fit(train, trainLabels)
    gnb_predictions = gnb.predict(test)
    tGnb = time.time() - t0
     
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, gnb_predictions), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Gaussian Naive Bayes")
    print("\n", classification_report(testLabels, gnb_predictions))
    print("Temps: " + str(tGnb) + "\n")
    
    
    print("Amb Features")
    gnb = GaussianNB()
    gnb.fit(trainHOG, trainLabels)
    gnb_predictionsHOG = gnb.predict(testHOG)
    tGnb = time.time() - t0
     
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, gnb_predictionsHOG), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Gaussian Naive Bayes + HOG Features")
    print("\n", classification_report(testLabels, gnb_predictionsHOG))
    print("Temps: " + str(tGnb) + "\n")
    

    #####################################################################################
    
    t0 = time.time()
    print("\n------ Model Decision Tree ------")
    
    print("Sense Features")
    dtree = DecisionTreeClassifier(max_depth = 2)
    dtree.fit(train, trainLabels)
    dtree_predictions = dtree.predict(test) 
    tDt = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, dtree_predictions), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Decision Tree")
    print("\n", classification_report(testLabels, dtree_predictions))
    print("Temps: " + str(tDt) + "\n")
    
    
    print("Amb Features")
    dtree = DecisionTreeClassifier(max_depth = 2)
    dtree.fit(trainHOG, trainLabels)
    dtree_predictionsHOG = dtree.predict(testHOG) 
    tDt = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, dtree_predictionsHOG), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Decision Tree + HOG Features")
    print("\n", classification_report(testLabels, dtree_predictionsHOG))
    print("Temps: " + str(tDt) + "\n")
    

    #####################################################################################
    
    t0 = time.time()
    print("\n------ Model Random Forest Classifier ------")
    
    print("Sense Features")
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(train, trainLabels)
    rf_predictions = rf.predict(test) 
    tRf = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, rf_predictions), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Random Forest Classifier")
    print("\n", classification_report(testLabels, rf_predictions))
    print("Temps: " + str(tRf) + "\n")


    print("Amb Features")
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(trainHOG, trainLabels)
    rf_predictionsHOG = rf.predict(testHOG) 
    tRf = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, rf_predictionsHOG), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Random Forest Classifier + HOG Features")
    print("\n", classification_report(testLabels, rf_predictionsHOG))
    print("Temps: " + str(tRf) + "\n")
    

    #####################################################################################
    
    t0 = time.time()
    print("\n------ Model Super Vector Machine (lineal) ------")
    
    print("Sense Features")
    svm = SVC(kernel = 'linear', C = 1)
    svm.fit(train, trainLabels)
    svm_predictions = svm.predict(test)
    tSvm = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, svm_predictions), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Super Vector Machine (lineal)")
    print("\n", classification_report(testLabels, svm_predictions))
    print("Temps: " + str(tSvm) + "\n")
    
    
    print("Amb Features")
    svm = SVC(kernel = 'linear', C = 1)
    svm.fit(trainHOG, trainLabels)
    svm_predictionsHOG = svm.predict(testHOG)
    tSvm = time.time() - t0
    
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_matrix(testLabels, svm_predictionsHOG), annot=True, cmap="OrRd")
    plt.title("Confusion Matrix Super Vector Machine (lineal) + HOG Features")
    print("\n", classification_report(testLabels, svm_predictionsHOG))
    print("Temps: " + str(tSvm) + "\n")
    
    #####################################################################################
    #listSignsDirectory = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5"]
    listSignsDirectory = ["signSobel_1", "signSobel_2", "signSobel_3", "signSobel_4", "signSobel_5"]
    #listSignsDirectory = ["signCF_1", "signCF_2", "signCF_3", "signCF_4", "signCF_5"]
    #listSignsDirectory = ["signCanny_1", "signCanny_2", "signCanny_3", "signCanny_4", "signCanny_5"]
    #listSignsDirectory = ["signOtsu_1", "signOtsu_2", "signOtsu_3", "signOtsu_4", "signOtsu_5"]
    train, trainLabels, test, testLabels = getTrainTestCNN(listSignsDirectory, 4, 0.8,cv2.COLOR_BGR2RGB)
    train = np.array(train)
    trainLabels = np.array(trainLabels)
    test = np.array(test)
    testLabels = np.array(testLabels)

    t0 = time.time()
    print("\n------ Model CNN ------")
    
    #Crear el modelo, este caso tendra 3 capas de 128 neuronas

    modelCNN = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), input_shape=[100, 150, 3], activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),#2,2 es el tamaño de la matriz
    
        tf.keras.layers.Conv2D(64, (3,3), input_shape=[100, 150, 3], activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),#2,2 es el tamaño de la matriz
    
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation=tf.nn.relu), #1a capa oculta activacion relu
        tf.keras.layers.Dense(128, activation=tf.nn.relu), #2a capa oculta activacion relu
        tf.keras.layers.Dense(128, activation=tf.nn.relu), #2a capa oculta activacion relu
        tf.keras.layers.Dense(5, activation=tf.nn.softmax), #capa de salida 15 salidas posibles
    ])
    modelCNN.compile(
        optimizer = 'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    modelCNN.summary()
    historialCNN = modelCNN.fit(train, trainLabels, epochs = 50)
    predictions = modelCNN.predict(test, verbose=False)
    
    #Evaluar nuestro modelo ya entrenado, contra el dataset de pruebas
    print("Resultado en las pruebas CNN con OHE: ")
    test_loss, test_accuracy = modelCNN.evaluate(test, testLabels)


    tSvm = time.time() - t0
    tablaAciertos(predictions, testLabels, test)
    print("Temps: " + str(tSvm) + "\n")
    
    #####################################################################################
    
    