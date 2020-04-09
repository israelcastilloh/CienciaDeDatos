#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:25:58 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#%% Importar los datos de digitos
digits = datasets.load_digits()

#%% Visualizar un dataset de digitos 
ndig = 10
for k in np.arange(ndig):
    plt.subplot(2,ndig/2,k+1)
    plt.imshow(digits.images[k], cmap=plt.cm.gray_r)
    plt.axis('off')
    plt.title('Digits: %i' %k)
plt.show()

#%% Aplicar el algoritmo de PCA en toda la base de datos
data = digits.data
media = data.mean(axis=0)
data_m = data-media
M_cov = np.cov(data_m, rowvar=False)
w, v = np.linalg.eig(M_cov)

#%%
porcentaje = w/np.sum(w)
porcentaje_acum = np.cumsum(porcentaje)

limite = 0.95
plt.bar(np.arange(len(porcentaje)), porcentaje)
plt.show()
plt.bar(np.arange(len(porcentaje_acum)), porcentaje_acum)
plt.hlines(limite, 0, 64, 'r')
plt.show()


#%% Proyectar los datos en las nuevas dimensiones
indx = porcentaje_acum<=limite
componentes = w[indx]
M_trans = v[:,indx]

data_new = np.array(np.matrix(data_m)*np.matrix(M_trans)) #el importante

#%% Recuperar las imagenes a partir de las variables educidas 
data_r = np.array(np.matrix(data_new)*np.matrix(M_trans.transpose()))
data_r = data_r + media
data_r[data_r<0] = 0

ndig = 10
for k in np.arange(ndig):
    plt.subplot(2,ndig/2,k+1)
    plt.imshow(np.reshape(data_r[k,:],(8,8)), cmap=plt.cm.gray_r)
    plt.axis('off')
    plt.title('Digits: %i' %k)
plt.show()




#%% Si decidimos Reducirlo a 2 variables
M_trans = v[:, 0:2]
data_new = np.array(np.matrix(data_m)*np.matrix(M_trans))
plt.scatter(data_new[:,0], data_new[:,1], c = digits.target)
plt.colorbar()
plt.grid()
plt.show()

#%% Reducir las variables por metodo de varianza
varianza = np.var(data, axis=0)
plt.bar(np.arange(len(varianza)), varianza)
plt.show()

#%% seleccionar el nivel de varianza
nivel_varianza = 0
idnx = varianza > nivel_varianza

data_new = data[:,idnx]
img_f = np. zeros(64)
img_f[idnx] = 15
img_f = np.reshape(img_f, (8,8))
plt.imshow(img_f, cmap=plt.cm.gray_r)
plt.show() #el de varianza nos dice que pixeles tomar de la imagen, pero tu lo propones










