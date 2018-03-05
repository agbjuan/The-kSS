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

def centroides(cdim, cren, clas, entrenamiento, prot):
    for i in range(len(clas)):
        c = 0
        for r in range(cren):
            if clas[i] == entrenamiento[r][cdim-3]:
                c += 1
                for j in range(cdim-3):
                    prot[i][j] += entrenamiento[r][j]
        for j in range(cdim-3):
            prot[i][j] /= c

#Cargar base de datos
train = []
with open("adult_train.dat", "r") as f2:
    reader = csv.reader(f2)
    for i in reader:
        train.append(i)

test = []
with open("adult_test.dat", "r") as f1:
    reader = csv.reader(f1)
    for i in reader:
        test.append(i)

train = [[int(a) for a in x] for x in train]
test = [[int(a) for a in x] for x in test]

datos = np.loadtxt("adult.dat", delimiter = ",")

print datos.shape
print datos
print

features = datos[:,0:-1]
targets = datos[:,-1]

folds = 10
k = -1

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

        #Se generan los centroides de los datos de entrenamiento
        prot = np.zeros((len(clas), cdim-1))
        for i in range(len(clas)):
            prot[i][cdim-3] = clas[i]

        centroides(cdim, cren, clas, entrenamiento, prot)

        #Calcular masa de los datos de entrenamiento
        for i in range(cren):
            #Identificacion del centroide de la misma clase y calcular distancia
            for j in range(len(prot)):
                if entrenamiento[i][cdim-3] == prot[j][cdim-3]:
                    d1 = distance.euclidean(entrenamiento[i, 0:cdim-3], prot[j, 0:cdim-3])
                    break

            #Distancia Euclidiana entre el objeto de train actual y los centroides
            for j in range(len(prot)):            
                dist = distance.euclidean(entrenamiento[i, 0:cdim-3], prot[j, 0:cdim-3])
                prot[j][cdim-2] = dist
                
            # Odenamiento de los centroides, por la distancia
            prot = prot[prot[:,cdim-2].argsort()]
            
            #Identifica el centroide de otra clase mas cercano y calcula la masa
            for j in range(len(prot)):
                if entrenamiento[i][cdim-3] != prot[j][cdim-3]:
                    d2 = prot[j][cdim-2]
                    break
           
            entrenamiento[i][cdim-2] = d1/d2
            #entrenamiento[i][cdim-2] = d2/d1

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
                
            # Encontrar la clase con mas elementos entre las fuerzas mas fuertes
            aux = 0
            for i in range(len(clas)):
                r = np.where(entrenamiento[:k,cdim-3] == clas[i])
                if len(r[0]) > aux:
                    c = clas[i]
                    aux = len(r[0])

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