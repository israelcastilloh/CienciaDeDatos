#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:22:24 2019

@author: israelcastilloh
"""


import pandas as pd
import numpy as np
import sklearn.metrics as skm
import scipy.spatial.distance as sc
from mylib import mylib

#%% IMPORTAR DATOS
accidents = pd.read_csv('../Data/Accidents_2015.csv') 

#%% APLICAR EL ESTUDIO DE CALIDAD DE DATOS
mireporte = mylib.dqr(accidents)

#%% ELEGIR LAS COLUMNAS PARA CONVERTIRLAS A DUMMIES
indx = np.array(accidents.dtypes == 'int64')
col_list = list(accidents.columns.values[indx])
accidents_int = accidents[col_list]

#%% VOLVER A APLICAR EL DQR
mireporte = mylib.dqr(accidents_int)

#%% TOMAR LAS COLUMNAS QUE TENGAN MENOS VALORES ÚNICOS
indx = np.array(mireporte['Valores unicos']<=10)
col_list_2 = np.array(col_list)[indx]
accidents_int_unique = accidents_int[col_list_2]

#%% OBTENER VARIABLES DUMMY DE UNA COLUMNA
dummy1= pd.get_dummies(accidents_int_unique['Accident_Severity'], 
                       prefix = 'Accident_Serverity')

#%% OBTENER VARIABLES DUMMY DE TODA LA TABLA
accidents_dummy = pd.get_dummies(accidents_int_unique[col_list_2[0]], 
                                 prefix = col_list_2[0])

for col in col_list_2[1:]:
    tmp = pd.get_dummies(accidents_int_unique[col],
                         prefix = col)
    accidents_dummy = accidents_dummy.join(tmp)

#%% APLICAR LOS INDICES DE SIMILITUD
D1 = sc.squareform(sc.pdist(accidents_dummy.iloc[0:30,:], 'matching'))

    