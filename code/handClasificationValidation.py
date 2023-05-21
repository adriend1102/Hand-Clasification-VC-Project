"""
Hand Clasification Validation: KNN - SVM lineal - CNN

"""

from handClasification import *

import pandas as pd

from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


def interpretScores(listScores, rangeValues, nameValues, title):
    
    prec = []
    recall = []
    f1 = []
    fitTime = []
    scoreTime = []
    
    for res in listScores:
        prec.append(np.mean(res['test_precision_weighted']))
        recall.append(np.mean(res['test_recall_weighted']))
        f1.append(np.mean(res['test_f1_weighted']))
        fitTime.append(np.mean(res['fit_time']))
        scoreTime.append(np.mean(res['score_time']))
    
    
    plt.plot(rangeValues, prec, label='Precision')
    plt.plot(rangeValues, recall, label='Recall')
    plt.plot(rangeValues, f1, label='F1-Score')
    plt.xlabel(nameValues)
    plt.ylabel('Metrics')
    plt.legend()
    plt.title(title)
    plt.show()

    plt.plot(rangeValues, fitTime)
    plt.xlabel(nameValues)
    plt.title(title)
    plt.ylabel('Fit Time')
    plt.show()
    
    plt.plot(rangeValues, scoreTime)
    plt.xlabel(nameValues)
    plt.ylabel('Score Time')
    plt.title(title)
    plt.show()
    
    

if __name__ == '__main__':
    
    # K Fold
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    
    """
    ##########################################################################################
    
    # Hiperparametros KNN
    print("------ Busqueda hiperparametros KNN + KFold ------")
    listSignsDirectory = ["signOtsu_1", "signOtsu_2", "signOtsu_3", "signOtsu_4", "signOtsu_5"]
    train, trainLabels, test, testLabels = getTrainTest(listSignsDirectory, 2, 0.8, cv2.IMREAD_GRAYSCALE)
    
    # Obtain features
    trainHOG, testHOG = hogDescriptor(train, test)  
    
    # Flatten train - test
    train = list(map(lambda x: cv2.resize(x, (x.shape[0], x.shape[1])).flatten(), train))
    test = list(map(lambda x: cv2.resize(x, (x.shape[0], x.shape[1])).flatten(), test))
    listScores = []
    listScoresHOG = []
    
    for k in range(2, 11):
        print("Calculating Knn with K = " + str(k) + "...")
        
        print("Sense Features")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train, trainLabels)
        listScores.append(cross_validate(knn, test, testLabels, cv=cv, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted', )))
        
        print("Amb HOG Features")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train, trainLabels)
        listScoresHOG.append(cross_validate(knn, testHOG, testLabels, cv=cv, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted', )))
    
    interpretScores(listScores, range(2, 11), 'Valors de K', 'Hiperparametres KNN')
    interpretScores(listScoresHOG, range(2, 11), 'Valors de K', 'Hiperparametres KNN amb HOG')
    
    
    ##########################################################################################
    """
    print("\n Metricas en diferentes divisiones train-test KNN + SVM")
    
    listScoresKNN = []
    listScoresKNN_Hog = []
    
    listScoresSVM = []
    listScoresSVM_Hog = []
    
    listScoresCNN = []
    
    # Probar models amb diferents percentatges de divisio
    percentatges = [0.5, 0.6, 0.7, 0.8]
    listSignsDirectory = ["signOtsu_1", "signOtsu_2", "signOtsu_3", "signOtsu_4", "signOtsu_5"]
    """
    for p in percentatges:
      
        print("Divisio " + str(int(p*100)) + "-" + str(int((1-p)*100)))
        train, trainLabels, test, testLabels = getTrainTest(listSignsDirectory, 2, p, cv2.IMREAD_GRAYSCALE)
        
        # Obtain features
        trainHOG, testHOG = hogDescriptor(train, test)
         
        # Flatten train - test
        train = list(map(lambda x: cv2.resize(x, (x.shape[0], x.shape[1])).flatten(), train))
        test = list(map(lambda x: cv2.resize(x, (x.shape[0], x.shape[1])).flatten(), test))
      
        ######################################################################################
        
        print("\n------ Model KNN ------")
        
        print("Sense Features")
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(train, trainLabels)
        listScoresKNN.append(cross_validate(knn, test, testLabels, cv=cv, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted', )))

        print("Amb HOG Features")
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(trainHOG, trainLabels)
        listScoresKNN_Hog.append(cross_validate(knn, testHOG, testLabels, cv=cv, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted', )))
        
        ######################################################################################
        
        print("\n------ Model Super Vector Machine (lineal) ------")
        
        print("Sense Features")
        svm = SVC(kernel = 'linear', C = 1)
        svm.fit(train, trainLabels)
        listScoresSVM.append(cross_validate(svm, test, testLabels, cv=cv, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted', )))

        print("Amb Features\n")
        svm = SVC(kernel = 'linear', C = 1)
        svm.fit(trainHOG, trainLabels)
        listScoresSVM_Hog.append(cross_validate(svm, testHOG, testLabels, cv=cv, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted', )))
          
        
    #########################################################################################
    
    interpretScores(listScoresKNN, percentatges, 'Percentatges de train', 'KNN')
    interpretScores(listScoresKNN_Hog, percentatges, 'Percentatges de train', 'KNN amb HOG')
    interpretScores(listScoresSVM, percentatges, 'Percentatges de train', 'SVM')
    interpretScores(listScoresSVM_Hog, percentatges, 'Percentatges de train', 'SVM amb HOG')
    
    """   
    ##########################################################################################

    print("\n Metricas en diferentes divisiones train-test CNN")
    
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
        
    
    for p in percentatges:
        
        print("\nDivisio " + str(int(p*100)) + "-" + str(int((1-p)*100)))
        print("------ Model CNN ------")
        
        train, trainLabels, test, testLabels = getTrainTestCNN(listSignsDirectory, 2, p, cv2.IMREAD_GRAYSCALE, 8)
        
        train = np.array(train)
        trainLabels = np.array(trainLabels)
        
        test = np.array(test)
        testLabels = np.array(testLabels)
     
        print("Sense Features\n")
        modelCNN.fit(train, trainLabels, epochs=50)
        
        # Define the KerasClassifier object and use it in cross_val_predict
        keras_clf = KerasClassifier(model = modelCNN, verbose=0)
        
        print("\nCross-Validate")
        listScoresCNN.append(cross_validate(keras_clf, test, testLabels, cv=cv, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted', )))
        
    interpretScores(listScoresCNN, percentatges, 'Percentatges de train', 'CNN')
