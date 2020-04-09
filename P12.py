#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:19:37 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt

#%%Datos originales o reales.

data = np.array([[2.5 , 2.4],
                 [0.5 , 0.7],
                 [2.2 , 2.9],
                 [3.1 , 3.0],
                 [1.8 , 2.1],
                 [1.1 , 1.1],
                 [1.5 , 1.6],
                 [1.1 , 0.9],
                 [4.0 , 4.1],
                 [0.1 , 0.2]])

plt.scatter(data[:,0],data[:,1])
plt.grid()
plt.plot()

#%% 1. Convertir los datos a un conjunto de media 0
media = data.mean(axis=0)
data_m = data - media
plt.scatter(data_m[:,0],data[:,1])
plt.grid()
plt.show()

#%% 2. Obtener matriz de covarianzas 
data_cov = np.cov(data_m, rowvar=False)

#%% 3. Calcular los valores (w) (importancia o magnitud) y vectores (v) (direccion) propios
w,v = np.linalg.eig(data_cov) 

#%% Dibujar los vectores propios como direcciones 
x = np.arange(-2,2,0.1)
plt.scatter(data_m[:,0], data_m[:,1])
plt.plot(x, (v[1,0]/v[0,0]*x), 'b--') #sacando la pendiente Vector propio 1
plt.plot(x, (v[1,1]/v[0,1]*x), 'g--') #sacando la pendiente Vector propio 2 
plt.axis('square')
plt.show()

#%% 4. Transformar los datos a los nuevos ejes
M_trans = v[:,[1,0]] #ordenando por importancialos vectores
compontentes = w[[1,0]] #ordenando por importancia los valores

data_new = np.array(np.matrix(data_m)*np.matrix(M_trans)) #transformar mis datos a los nuevos ejes
plt.scatter(data_new[:,0], data_new[:,1])
plt.axis('square')
plt.grid()
plt.show()
















