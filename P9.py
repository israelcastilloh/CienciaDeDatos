#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:20:02 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#%% Generar los datos para clustering

semilla = 1500
X,Y = make_blobs(n_samples=1000, random_state = semilla)

plt.scatter(X[:,0], X[:,1])
plt.show()

#%% Aplicar el KMeans
model = KMeans(n_clusters=3, random_state=semilla, init='random')

model = model.fit(X) #ejecuta el algoritmo

Ypredict = model.predict(X)
centroides = model.cluster_centers_
J = model.inertia_

#Visualizar los resultados
plt.scatter(X[:,0],X[:,1], c=Ypredict)
plt.plot(centroides[:,0], centroides[:,1], 'x')
plt.show()

#%% Criterio de decision del numero de grupos
#Grafica de codo

inercias = np.zeros(10) #preallocation
for k in np.arange(len(inercias))+1:
    model = KMeans(n_clusters=k, random_state=semilla, init='random')
    model = model.fit(X)
    inercias[k-1]= model.inertia_
    
plt.plot(np.arange(len(inercias))+1, inercias)
plt.xlabel('Num de grupos')
plt.ylabel('Inercia Total')
plt.show()