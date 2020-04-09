#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:44:58 2019

@author: israelcastilloh
"""



import pandas as pd
import numpy as np
import sklearn.metrics as skm
import scipy.spatial.distance as sc
from mylib import mylib
import matplotlib.pyplot as plt
import string 
from datetime import datetime
#%%

enfermedades = pd.read_csv("../Data/enfermedades.csv",encoding='latin1')

#%% Calculo de los rendimientos de casos de enfermedades
df = pd.DataFrame()
df['2010'] = enfermedades['2010']
df['2011'] = enfermedades['2011']
df['2012'] = enfermedades['2012']
df['2013'] = enfermedades['2013']
df['2014'] = enfermedades['2014']


rend1 = (df['2011']-df['2010'])/df['2010']
rend2 = (df['2012']-df['2011'])/df['2011']
rend3 = (df['2013']-df['2012'])/df['2012']
rend4 = (df['2014']-df['2013'])/df['2013']
#%% Calculo de los porcentajes.
resumen = pd.DataFrame()
resumen['2010'] = rend1.values
resumen['2011'] = rend2.values
resumen['2012'] = rend3.values
resumen['2013'] = rend4.values
resumen.rename(index=enfermedades.nombre, inplace = True)

resument= np.transpose(resumen)

#%% Indices de similitud
resument.plot()
plt.xlabel('Años')
plt.ylabel('Porcentaje')
plt.title('Cambio porcentual de las enfermedades')

plt.grid()


#%% Indices de similitud


D1 = sc.pdist(resument,'euclidean')
D1 = sc.squareform(D1)

D2 = sc.pdist(resumen,'euclidean')
D2 = sc.squareform(D2)

