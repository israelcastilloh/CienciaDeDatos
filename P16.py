#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:25:48 2019

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

#%% Preparación de los datos (crear el polinomio)
ngrado = 14
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(X)

#%% Crear y entrenar el modelo de clasificación
modelo = linear_model.LogisticRegression(C=1e20)
modelo.fit(Xa,Y)

Yhat = modelo.predict(Xa)

#%% Evaluar el desempeño
accuracy_score(Y,Yhat)

#%% Evaluar la precisión
precision_score(Y,Yhat)

#%%Evaluar el recall
recall_score(Y,Yhat)

#%% Visualizar la superficie 
h = 0.01
x0min,x0max,x1min,x1max = X[0].min(),X[0].max(),X[1].min(),X[1].max()
xx,yy = np.meshgrid(np.arange(x0min,x0max,h), np.arange(x1min,x1max,h))

Xnew = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
Xa_new = poly.fit_transform(Xnew)
Z = modelo.predict(Xa_new)
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
plt.scatter(X[0], X[1], c=Y)
plt.xlim(x0min,x0max)
plt.ylim(x1min,x1max)
plt.show()

#%% Observar el sobreajuste
W = modelo.coef_
plt.bar(np.arange(len(W[0])),W[0])
plt.show()



