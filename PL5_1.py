#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:32:00 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import pandas as pd


#%% Importar los datos
data = pd.read_excel('../Data/Datos_2015.xlsx',
                     sheetname = 'Atemajac')

#%%
data = data.iloc[:, 0:7].dropna()

#%% Dibujar un par de contaminantes
plt.scatter(data['CO'], data['NO2'])
plt.xlabel('CO'), plt.ylabel('NO2')
plt.axis('square')
plt.show()

plt.scatter(data['CO'], data['PM10'])
plt.xlabel('CO'), plt.ylabel('PM10')
plt.axis('square')
plt.show()

#%% APLICAR LA DISTANCIA EUCLIDEANA SIN NORMALIZAR
D1 = sc.squareform(sc.pdist(data.iloc[:,2:].T, 'euclidean'))

#%%
data = data.iloc[:,2:]
#%% NORMALIZAR LOS DATOS

data_norm=(data-data.mean(axis=0))/data.std(axis=0)

 #%%VISUALIZAR LOS DATOS
plt.subplot(121)
plt.scatter(data['CO'], data['PM10'])
plt.xlabel('CO'), plt.ylabel('PM10')
plt.axis('square')

plt.subplot(122)

plt.scatter(data_norm['CO'], data_norm['PM10'])
plt.xlabel('CO'), plt.ylabel('PM10')
plt.axis('square')
plt.show()

