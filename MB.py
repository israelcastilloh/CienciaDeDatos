#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:27:19 2019

@author: israelcastilloh
"""


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
import pickle 

#%% Importar los datos
data = pd.read_csv('../Data/Multiburo.csv')
X = data.iloc[:,1:30]
Y = data.iloc[:,30]

#del data

#%%Normalizar Amount
X['Amount']=(X['Amount']-X['Amount'].mean())/X['Amount'].std()

#%%Seleccionar los datos de entrenamiento y prueba 
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.5,
                                                    random_state=0)

del X,Y

#%%REGRESION LOGISTICA (REVISAR EL POLINOMIO OPTIMO)
modelo_rl = linear_model.LogisticRegression()
grado = np.arange(1,3)
Accu = np.zeros(grado.shape)
Prec = np.zeros(grado.shape)
Reca = np.zeros(grado.shape)

for ngrado in grado:
    poly = PolynomialFeatures(ngrado)
    Xa = poly.fit_transform(X_train)
    modelo_rl.fit(Xa,Y_train)
    Yhat =modelo_rl.predict(Xa)
    Accu[ngrado-1] = accuracy_score(Y_train, Yhat)
    Prec[ngrado-1] = precision_score(Y_train, Yhat)
    Reca[ngrado-1] = recall_score(Y_train, Yhat)

#%%Graficar el codo

plt.plot(grado, Accu)
plt.plot(grado, Prec)
plt.plot(grado, Reca)
plt.xlabel('Grado de polinomio')
plt.ylabel('% de Aciertos')
plt.legend(('Accu', 'Prec', 'Reca'), loc='best')
plt.grid()
plt.show()

#%%
ngrado = 2
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X_train)
modelo_rl.fit(Xa,Y_train)
Yhat = modelo_rl.predict(Xa)
print(accuracy_score(Y_train, Yhat))
print(precision_score(Y_train, Yhat))
print(recall_score(Y_train, Yhat))

#%% Probar con datos desconocidos
Xa_test = poly.fit_transform(X_test)
Yhat_rl_test = modelo_rl.predict(Xa_test)
print(accuracy_score(Y_test, Yhat_rl_test))
print(precision_score(Y_test, Yhat_rl_test))
print(recall_score(Y_test, Yhat_rl_test))

#%% PROBAR EL MODELO SVM
modelo_sm = svm.SVC(kernel='rbf')
modelo_svm.fit(X_train, Y_train)
Yhat_sv_train = modelo_svm.predict(X_train)
print(accuracy_score(Y_train, Yhat_sv_test))
print(precision_score(Y_train, Yhat_sv_test))
print(recall_score(Y_train, Yhat_sv_test)) 
    
#%% Evaluar los datos de test con SVM
Yhat_sv_test = modelo_svm.predict(X_test)
print(accuracy_score(Y_test, Yhat_sv_test))
print(precision_score(Y_test, Yhat_sv_test))
print(recall_score(Y_test, Yhat_sv_test))

    