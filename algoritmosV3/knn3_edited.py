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
with open("pima_train.dat", "r") as f2:
    reader = csv.reader(f2)
    for i in reader:
        train.append(i)

test = []
with open("pima_test.dat", "r") as f1:
    reader = csv.reader(f1)
    for i in reader:
        test.append(i)

train = [[int(a) for a in x] for x in train]
test = [[int(a) for a in x] for x in test]

datos = np.loadtxt("pima.dat", delimiter = ",")

print datos.shape
#print datos
print

features = datos[:,0:-1]
targets = datos[:,-1]

folds = 10
k = -1
k2 = 7

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
            # Distancia Euclidiana entre el objeto de train actual y los demas datos de train
            for i in range(cren):
                dist = distance.euclidean(entrenamiento[t, 0:cdim-3], entrenamiento_copia[i, 0:cdim-3])
                entrenamiento_copia[i][cdim-1] = dist
            
            #Ordenamiento de los datos por distancia
            entrenamiento_copia = entrenamiento_copia[entrenamiento_copia[:,cdim-1].argsort()]

            #Contar k2 vecinos de la misma clase que el dato de train actual
            r = np.where(entrenamiento_copia[1:k2+1,cdim-3] == entrenamiento[t,cdim-3])
            r = len(r[0])
            entrenamiento[t][cdim-2] = (r+1)*1.0 / (k2+1)
            #entrenamiento[t][cdim-2] = (k2+1-r)*1.0 / (k2+1)
                
        entrenamiento_copia = []

        resultados = []
        for t in range(cren2):
            #print pruebas[t]
            #print
            
            # Fuerza entre el objeto a clasificar y los datos de train
            for i in range(cren):
                dist = distance.euclidean(entrenamiento[i, 0:cdim-3], pruebas[t, 0:cdim-3])
                entrenamiento[i][cdim-1] = entrenamiento[i][cdim-2]*pruebas[t][cdim-3]/math.pow(dist, 2)

            #print entrenamiento
            #print
         
            # Ordenamiento de los vecinos, por la fuerza, de mayor a menor
            entrenamiento = entrenamiento[entrenamiento[:,cdim-1].argsort()[::-1]]

            #print entrenamiento
            #print
            
            # Obtener las diferentes clases que existen entre las k fuerzas mas fuertes
            clas = []
            for i in range(k):
                if entrenamiento[i][cdim-3] not in clas:
                    clas.append(entrenamiento[i][cdim-3])

            #print clas
            #print
                
            # Encontrar la clase con mas fuerza acumulada entre las fuerzas mas fuertes
            aux = 0
            for i in range(len(clas)):
                r = np.where(entrenamiento[:k,cdim-3] == clas[i])
                w = np.sum(entrenamiento[ r, cdim-1])
                if w > aux:
                    c = clas[i]
                    aux = w

            #print c
            #print
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