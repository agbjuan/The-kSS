import numpy as np
import math
import pandas
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from scipy.spatial import distance
import os
import csv

#Cargar base de datos
train = []
with open("glass_train.dat", "r") as f2:
    reader = csv.reader(f2)
    for i in reader:
        train.append(i)

test = []
with open("glass_test.dat", "r") as f1:
    reader = csv.reader(f1)
    for i in reader:
        test.append(i)

train = [[int(a) for a in x] for x in train]
test = [[int(a) for a in x] for x in test]

datos = np.loadtxt("glass.dat", delimiter = ",")

print datos.shape
print datos
print

features = datos[:,0:-1]
targets = datos[:,-1]

#Reescalar entre 0-1
#min_max_scaler = preprocessing.MinMaxScaler()
#features = min_max_scaler.fit_transform(features)

#Normalizar, L1 o L2
#features = preprocessing.normalize(features, norm = 'l2')

folds = 10
k = -1
k2 = 5
w = 10

#for k in range(1, barrido + 1):
for z in range(4):
    k += 2
    acc_sum = 0
    fscore_sum = 0
    for ite in range(folds):
        x_train, x_test = features[train[ite]], features[test[ite]]
        y_train, y_test = targets[train[ite]], targets[test[ite]]
        
        y_train = y_train.tolist()
        y_train = np.array([y_train]).T

        #####
        entrenamiento = np.concatenate((x_train, y_train), axis = 1)
        zeros = np.zeros((len(entrenamiento), 2))
        entrenamiento = np.concatenate((entrenamiento, zeros), axis = 1)
        pruebas = x_test
        zeros = np.zeros((len(pruebas), 1))
        for i in zeros:
            i += 1
        pruebas = np.concatenate((pruebas, zeros), axis = 1)
        #####

        cren = len(entrenamiento)
        cdim = len(entrenamiento[0])
        cren2 = len(pruebas)

        #Se guarda una lista de las clases de los datos de entrenamiento
        clas = []
        for i in range(cren):
            if entrenamiento[i][cdim-3] not in clas:
                clas.append(entrenamiento[i][cdim-3])

        #Calcular masa de los datos de entrenamiento
        entrenamiento_copia = np.copy(entrenamiento)
        for t in range(cren):
            # Distancia Euclidiana enrte el objeto de train actual y los demas datos de train
            for i in range(cren):
                dist = distance.euclidean(entrenamiento[t, 0:cdim-3], entrenamiento_copia[i, 0:cdim-3])
                entrenamiento_copia[i][cdim-1] = dist
            
            #Ordenamiento de los datos por distancia
            entrenamiento_copia = entrenamiento_copia[entrenamiento_copia[:,cdim-1].argsort()]
            
            #Suma/resta de masa al objeto
            for i in range(1, k2+1):
                if entrenamiento[t][cdim-3] == entrenamiento_copia[i][cdim-3]:
                    entrenamiento[t][cdim-2] += w
                else:
                    entrenamiento[t][cdim-2] -= w

        entrenamiento_copia = []

        resultados = []
        for t in range(cren2):
            #print pruebas[t]
            #print

            # Distancia Euclidiana entre el objeto a clasificar y los datos de train
            for i in range(cren):
                dist = distance.euclidean(pruebas[t, 0:cdim-3], entrenamiento[i, 0:cdim-3])
                entrenamiento[i][cdim-1] = dist

            #print entrenamiento
            #print
         
            # Ordenamiento de los vecinos, por la distancia
            entrenamiento = entrenamiento[entrenamiento[:,cdim-1].argsort()]

            #print entrenamiento
            #print

            # Obtener las diferentes clases que existen entre los k vecinos mas cercanos
            clas = []
            for i in range(k):
                if entrenamiento[i][cdim-3] not in clas:
                    clas.append(entrenamiento[i][cdim-3])

            #print clasificaciones
            #print
                
            # Calculo de las fuerzas totales entre cada clase y el objeto a clasificar
            fuerzas = []
            for i in range(len(clas)):
                f = 0
                for j in range(k):
                    if clas[i] == entrenamiento[j][cdim-3]:
                        if entrenamiento[j][cdim-1] != 0:
                            f += entrenamiento[j][cdim-2]*pruebas[t][cdim-3]/math.pow(entrenamiento[j][cdim-1], 2)
                fuerzas.append(f)

            #print fuerzas
            #print
                
            # Prediccion de la clase
            c = clas[fuerzas.index(max(fuerzas))]
            
            #print c
            resultados.append(c)
            #print "////////////////////////////////////////////////////"

        #print resultados
        #print y_test

        acc_sum += accuracy_score(y_test, resultados)
        fscore_sum += f1_score(y_test, resultados, average = "macro")

    acc_actual = acc_sum/folds
    fscore_actual = fscore_sum/folds

    print
    print "Vecinos considerados: ", k
    print "Accuracy (avr): ", acc_actual
    print "F-Score (avr): ", fscore_actual
    print