#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:35:10 2019

@author: israelcastilloh
"""

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans


#%% Descargar los datos
inicio = datetime(2018,3,12)
final = datetime(2019,3,12)
#data = web.YahooDailyReader(symbols='CC', start=inicio, end=final, interval ='d').read()
data = get_nasdaq_symbols ()

#%%Visualizar la serie de datos
plt.plot(data['Close'])
plt.show()

#%% Descompener la serie original en subseries de lingutd nv
dat = data['Close']
nv = 5
n_prices = len(dat)
dat_new = np.zeros((n_prices-nv,nv))
for k in np.arange(nv):
    dat_new[:,k]=dat[k:(n_prices-nv)+k]
    
    
    
    
#%% Normalizar los datos
tmp = dat_new.transpose()
tmp = (tmp - tmp.mean(axis=0))/tmp.std(axis=0)
dat_new = tmp.transpose()   
    
#%% Ver las ventanas
plt.plot(dat_new.T)
plt.xlabel('time')
plt.ylabel('prices')
plt.show()

#%%Buscar patrones con el algoritmo de clustering
n_clusters = 15
inercias = np.arange(n_clusters)
for k in np.arange(n_clusters)+1:
    model = KMeans(n_clusters=k, init='k-means++').fit(dat_new) ##inicializa los clusters lejos
    inercias[k-1] = model.inertia_

#%%
plt.plot(np.arange(n_clusters)+1, inercias)
plt.plot(np.arange(len(inercias))+1, inercias)
plt.xlabel('# grupos')
plt.ylabel('Inercia Total')
plt.show()

#%% Generar los clusters
model = KMeans(n_clusters=5, init='k-means++').fit(dat_new)
grupos = model.predict(dat_new)
centroides = model.cluster_centers_

plt.plot(centroides.T)
plt.show()

#%% Dibujar todos los centroides por separado
n_subfig = np.ceil(np.sqrt(len(np.unique(grupos))))
for k in np.unique(grupos):
    plt.subplot(n_subfig, n_subfig, k+1)
    plt.plot(centroides[k,:])
    plt.ylabel('Grupo %d' %k)
    
plt.show()

#%%
plt.subplot(211)
plt.plot(dat)
plt.xlabel('time')
plt.ylabel('price')
plt.subplot(212)
plt.bar(np.arange(nv,len(dat)),grupos)
plt.xlabel('time')
plt.ylabel('grupo')
plt.show()

















    
    
    
    
    
    
    
    
    
    