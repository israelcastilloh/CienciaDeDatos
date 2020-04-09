#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:52:34 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score)

#%% Importar los datos
data = pd.read_csv('../Data/ex2data2.txt', header=None)

X = data.iloc[:,0:2]
Y = data.iloc[:,2]

plt.scatter(X[0],X[1],c=Y)
plt.show()

#%% Buscar el grao del polinomio optimo
modelo = linear_model.LogisticRegression() #si omitimos, C=1
grados = np.arange(1,20)
Accu = np.zeros(grados.shape)
Prec = np.zeros(grados.shape)
Rec = np.zeros(grados.shape)
nW = np.zeros(grados.shape)

for ngrado in grados:
    poly = PolynomialFeatures(ngrado)
    Xa = poly.fit_transform(X)
    modelo.fit(Xa,Y)
    Yhat = modelo.predict(Xa)
    Accu[ngrado-1] = accuracy_score(Y,Yhat)
    Prec[ngrado-1] = precision_score(Y,Yhat)
    Rec[ngrado-1] = recall_score(Y,Yhat)
    nW[ngrado-1] = len(modelo.coef_)

plt.plot(grados,Accu)
plt.plot(grados,Prec)
plt.plot(grados,Rec)
plt.xlabel('grado polinomio')
plt.ylabel('% aciertos')
plt.legend(('Accuracy', 'Precision', 'Recall'), loc='best')
plt.grid()
plt.show()

#%% Seleccionar el modelo deseado
ngrado = 5
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)
modelo = linear_model.LogisticRegression()
modelo.fit(Xa,Y)
Yhat = modelo.predict(Xa)

#%%Optimización del modelo
#Seleccionar parametros > umbral
W = modelo.coef_[0]
plt.bar(np.arange(len(W)), W)
plt.grid()
plt.show()

#%%
umbral = 0.5
indx = np.abs(W)>umbral
Xa_simplificada = Xa[:,indx]
modelo_opt = linear_model.LogisticRegression()
modelo_opt.fit(Xa_simplificada,Y)
Yhat_opt = modelo_opt.predict(Xa_simplificada)

#%%
accuracy_score(Y,Yhat)

#%%
accuracy_score(Y,Yhat_opt)









