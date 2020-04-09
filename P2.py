#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:15:01 2019

@author: israelcastilloh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.metrics as skm
import scipy.spatial.distance as sc
 
#%% importar los datos de analisis

digits =  datasets.load_digits()


#%% dibujar una muestra de los digitos

ndig = 10
for k in np.arange(ndig):
    plt.subplot(2, ndig/2, k+1)
    plt.axis('off')
    plt.imshow(digits.images[k], cmap = plt.cm.gray_r)
    plt.title('Digit: %i' % k)
plt.show()


#%% seleccionar una muestra de datos y convertirrla

data = digits.data[0:30]
umbral  = 7
data[data<=umbral] = 0
data[data>umbral] = 1
data = pd.DataFrame(data)

#%% calcular indices de similitud
cf_m = skm.confusion_matrix(data.iloc[0,:], data.iloc[10,:])
m_simple = skm.accuracy_score(data.iloc[0,:], data.iloc[10,:])
sim_simple_manual = (cf_m[0,0]+cf_m[1,1])/np.sum(cf_m) #suma de diagonales / la suma total

#sim_jac = skm.jaccard_similarity_score(data.iloc[0,:], data.iloc[1,:])
sim_jac_manual = (cf_m[1,1])/(cf_m[0,1]+cf_m[1,0]+cf_m[1,1])

#%% calcular las distancias para binarios
d1 = sc.matching(data.iloc[0,:],data.iloc[10,:])
d2 = sc.jaccard(data.iloc[0,:],data.iloc[10,:])

#%% calcular todas las combinaciones posibles
D1 = sc.pdist(data, 'matching')
D1 = sc.squareform(D1)









