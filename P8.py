#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:14:53 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import scipy.spatial.distance as sc

#%% Generar los datos para el clustering
np.random.seed(4711)
a = np.random.multivariate_normal([10,0], [[3,0], [0,3]], size = [100])
b = np.random.multivariate_normal([0,20], [[3,0], [0,3]], size = [100])
c = np.random.multivariate_normal([3,7], [[3,0], [0,3]], size = [100])
# se tiene le centro del conjunto, la varianza y covarianza de una y otra

X = np.concatenate((a,b,c),)
plt.scatter(X[:,0],X[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('squared')
plt.grid()
plt.show()

#%% Aplicar el algoritmo de clustering
Z = hierarchy.linkage(X, metric='euclidean', method = 'single') #correlation, ward

plt.figure(figsize = (25,15))
plt.title('dendrograma completo')
plt.xlabel('Indice de la muestra')
plt.ylabel('Distancia o Similitud')
dn = hierarchy.dendrogram(Z)
plt.show()

#%% Mostrar un conjunto de datos ejemplos
idx = [40]
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.scatter(X[idx,0],X[idx,1], c='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('square')
plt.grid()
plt.show()

#%% Modificar el aspecto dendograma
plt.figure()
plt.title('Dendograma completo')
plt.xlabel('Indice de muestra')
plt.ylabel('Distancia p Similitud')
dn = hierarchy.dendrogram(Z, 
                         truncate_mode = 'lastp', #level
                         p = 4)
plt.show()


#%% Criterio de codo (Parte 1)
last = Z[-15:,2]
last_rev = last[::-1]
idxs = np.arange(1,len(last_rev)+1)
plt.plot(idxs, last_rev)
plt.xlabel('# grupos')
plt.ylabel('distancia equivalente')
plt.grid()
plt.show()

#%% Criterio del gradiente (parte 2)
gradiente = -np.diff(last_rev)
plt.plot(idxs[1:], gradiente)
plt.xlabel('#grupos')
plt.ylabel('gradiente de distancia')  
plt.grid()
plt.show()

         
#%% Seleccionar elementos de los grupos formados
gruposmax = 10 # maximo numero de clusters, 
grupos = hierarchy.fcluster(Z, gruposmax, criterion = 'maxclust') #en que elementos pertenece a que grupo

plt.figure()
plt.scatter(X[:,0], X[:,1], c=grupos,
            cmap = plt.cm.Accent)
plt.show()

#%% Seleccionar elementos de los grupos formados
distmax = 1 #que cada grupo tenga dsitancia maxima de .90 o un velo decidido
grupos = hierarchy.fcluster(Z, distmax, criterion = 'distance') #en que elementos pertenece a que grupo

plt.figure()
plt.scatter(X[:,0], X[:,1], c=grupos,
            cmap = plt.cm.Accent)
plt.show()

#%% Seleccionar solo datos de un grupo

idx = grupos==5
subdata = X[idx,:]