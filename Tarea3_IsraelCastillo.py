#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:50:45 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import pandas as pd
from mylib import mylib


#%% Importar los datos
data1 = pd.read_excel('../Data/Datos_2015.xlsx', sheetname='Centro')
data2 = pd.read_excel('../Data/Datos_2015.xlsx', sheetname='Atemajac')
data3 = pd.read_excel('../Data/Datos_2015.xlsx', sheetname='Tlaquepaque')
data4 = pd.read_excel('../Data/Datos_2015.xlsx', sheetname='aguilas')
#%%
CO1 = pd.DataFrame(data1['CO'])
CO1.columns = ['CO1']
CO2 = pd.DataFrame(data2['CO'])
CO2.columns = ['CO2']
CO3 = pd.DataFrame(data3['CO'])
CO3.columns = ['CO3']
CO4 = pd.DataFrame(data4['CO'])
CO4.columns = ['CO4']

CO_total = CO1.join(CO2).join(CO3).join(CO4).dropna()
mireporte1 = mylib.dqr(CO_total)

#%% Aplicar la distancia euclideana sin normalizar
D1 = sc.squareform(sc.pdist(CO_total.T,'euclidean'))

#%%
PM10_1 = pd.DataFrame(data1['PM10'])
PM10_1.columns = ['PM10_1']
PM10_2 = pd.DataFrame(data2['PM10'])
PM10_2.columns = ['PM10_2']
PM10_3 = pd.DataFrame(data3['PM10'])
PM10_3.columns = ['PM10_3']
PM10_4 = pd.DataFrame(data4['PM10'])
PM10_4.columns = ['PM10_4']

PM10_total = PM10_1.join(PM10_2).join(PM10_3).join(PM10_4).dropna()
mireporte2 = mylib.dqr(PM10_total)

#%% Aplicar la distancia euclideana sin normalizar
D2 = sc.squareform(sc.pdist(PM10_total.T,'euclidean'))