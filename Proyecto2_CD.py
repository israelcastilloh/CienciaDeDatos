#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:37:47 2019

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
from sklearn import datasets


#%% Importar los datos

data = pd.read_csv('../Data/Audit.csv')
mireporte = mylib.dqr(data)
data= data.drop(['LOCATION_ID'],axis=1)
data = data.dropna()
#%%
X = data.iloc[:,:26]
Y = data.iloc[:,26]

#%%Seleccionar los datos de entrenamiento y prueba 
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.5,
                                                    random_state=0)

#%%REGRESION LOGISTICA (REVISAR EL POLINOMIO OPTIMO)
modelo_rl = linear_model.LogisticRegression()
grado = np.arange(1,5)
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

#%% Probar con datos desconocidos Modelo Logístico -Test-
Xa_test = poly.fit_transform(X_test)
Yhat_rl_test = modelo_rl.predict(Xa_test)
print(accuracy_score(Y_test, Yhat_rl_test))
print(precision_score(Y_test, Yhat_rl_test))
print(recall_score(Y_test, Yhat_rl_test))

#%% PROBAR EL MODELO SVM Entrenados
modelo_svm = svm.SVC(kernel='rbf')
modelo_svm.fit(X_train, Y_train)
Yhat_sv_train = modelo_svm.predict(X_train)
print(accuracy_score(Y_train, Yhat_sv_train))
print(precision_score(Y_train, Yhat_sv_train))
print(recall_score(Y_train, Yhat_sv_train)) 
    
#%% Evaluar los datos de test con SVM -Test-
Yhat_sv_test = modelo_svm.predict(X_test)
print(accuracy_score(Y_test, Yhat_sv_test))
print(precision_score(Y_test, Yhat_sv_test))
print(recall_score(Y_test, Yhat_sv_test))



#%%  PCA en toda la base de datos y Proyectar los datos en las nuevas dimensiones
#Obtener matriz M_trans
media = X.mean(axis=0)
data_m = X-media
M_cov = np.cov(data_m, rowvar=False)
w, v = np.linalg.eig(M_cov)
    
porcentaje = w/np.sum(w)
porcentaje_acum = np.cumsum(porcentaje)

limite = 0.95
plt.bar(np.arange(len(porcentaje)), porcentaje)
plt.show()
plt.bar(np.arange(len(porcentaje_acum)), porcentaje_acum)
plt.hlines(limite, 0, 27, 'r')
plt.show()

indx = porcentaje_acum<=limite
componentes = w[indx]
M_trans = v[:,indx]

#%% Si decidimos Reducirlo a 3 variables
x= 15
M_trans = v[:, 0:x]
Xtrain_new = np.array(np.matrix(X_train-media)*np.matrix(M_trans))  #el importante con train after PCA
Xtest_new = np.array(np.matrix(X_test-media)*np.matrix(M_trans)) #la matriz PCA de Xtest
plt.scatter(Xtrain_new[:,0], Xtrain_new[:,x-1])
plt.colorbar()
plt.grid()
plt.show()





#%% Modelo Logístico con 3 variables
X = Xtrain_new
Y = Y_train

#%%2da REGRESION LOGISTICA (REVISAR EL POLINOMIO OPTIMO)
modelo_rl = linear_model.LogisticRegression()
grado = np.arange(1,5)
Accu = np.zeros(grado.shape)
Prec = np.zeros(grado.shape)
Reca = np.zeros(grado.shape)

for ngrado in grado:
    poly = PolynomialFeatures(ngrado)
    Xa = poly.fit_transform(Xtrain_new)
    modelo_rl.fit(Xa,Y_train)
    Yhat =modelo_rl.predict(Xa)
    Accu[ngrado-1] = accuracy_score(Y_train, Yhat)
    Prec[ngrado-1] = precision_score(Y_train, Yhat)
    Reca[ngrado-1] = recall_score(Y_train, Yhat)
    
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
Xa = poly.fit_transform(Xtrain_new)
modelo_rl.fit(Xa,Y_train)
Yhat = modelo_rl.predict(Xa)
print(accuracy_score(Y_train, Yhat))
print(precision_score(Y_train, Yhat))
print(recall_score(Y_train, Yhat))

#%% Probar con datos desconocidos Modelo Logístico -Test-
Xa_test_new = poly.fit_transform(Xtest_new)
Yhat_rl_test = modelo_rl.predict(Xa_test_new)
print(accuracy_score(Y_test, Yhat_rl_test))
print(precision_score(Y_test, Yhat_rl_test))
print(recall_score(Y_test, Yhat_rl_test))


#%% PROBAR EL MODELO SVM Entrenados -Train-
modelo_svm = svm.SVC(kernel='rbf')
modelo_svm.fit(Xtrain_new, Y_train)
Yhat_sv_train = modelo_svm.predict(Xtrain_new)
print(accuracy_score(Y_train, Yhat_sv_train))
print(precision_score(Y_train, Yhat_sv_train))
print(recall_score(Y_train, Yhat_sv_train)) 
    
#%% Evaluar los datos de test con SVM -Test-
Yhat_sv_test = modelo_svm.predict(Xtest_new)
print(accuracy_score(Y_test, Yhat_sv_test))
print(precision_score(Y_test, Yhat_sv_test))
print(recall_score(Y_test, Yhat_sv_test))










