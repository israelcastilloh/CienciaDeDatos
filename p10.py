#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:12:29 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#%%Importar los datos 
data = pd.read_csv('../Data/creditcard.csv')
data = data.drop(['Time', 'Class'], axis = 1)

#%%Aplicar algoritmo de clustering
#Grafica de codo

 inercias = np.zeros(10) #preallocation
for k in np.arange(len(inercias))+1:
    model = KMeans(n_clusters=k, init='random')
    model = model.fit(data)
    inercias[k-1]= model.inertia_
    
plt.plot(np.arange(len(inercias))+1, inercias)
plt.xlabel('Num de grupos')
plt.ylabel('Inercia Total')
plt.show()

#%%Normalizar la columna Amount
data.Amount = (data.Amount-data.Amount.mean())/data.Amount.std()

#%% Clasificar los datos segun las graficas de codo
model = KMeans(n_clusters=4, init='random')
model = model.fit(data)
grupos = model.predict(data)
 
gg#%%Graficar los centroides
centroides = model.cluster_centers_
plt.plot(centroides.transpose())
plt.grid()
plt.show()
