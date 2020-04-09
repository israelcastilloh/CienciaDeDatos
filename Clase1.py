#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:08:17 2019

@author: israelcastilloh
"""

### DATA QUALITY REPORT 


import pandas as pd
import numpy as np

#%% Leer datos
accidents = pd.read_csv('../Data/Accidents_2015.csv')
    #accidents = pd.read_csv('Accidents_2015.csv')


#%% Determinar el nombre de las variables
columns = pd.DataFrame(list(accidents.columns.values), 
                       columns=['Nombres'],
                       index = list(accidents.columns.values))


#%% Determinar el tipo de variable
d_types = pd.DataFrame(accidents.dtypes,
                       columns=['Tipo'])

#%% Determinar valores perdidos
missing_values = pd.DataFrame(accidents.isnull().sum(axis=0),
                              columns=['Valores perdidos'])

#%% Determinar valores presentes
present_values = pd.DataFrame(accidents.count(),
                              columns=['Valores presentes'])

#%% Determinar valores únicos
unique_values = pd.DataFrame(accidents.nunique(),
                              columns=['Valores unicos'])

#%% Valores minimos y maximos de cada variable
min_values = pd.DataFrame(columns=['Min'])
max_values = pd.DataFrame(columns=['Max'])
for col in list(accidents.columns.values):
    try:
        min_values.loc[col]=[accidents[col].min()]
        max_values.loc[col]=[accidents[col].max()]
    except: 
        pass

#%% Juntar todas las tablas DATA QUALITY REPORT
dqr = columns.join(d_types).join(missing_values).join(
        present_values).join(unique_values).join(min_values).join(max_values)

