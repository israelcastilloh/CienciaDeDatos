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
casualties = pd.read_csv('../Data/Casualties_2015.csv')
vehicles = pd.read_csv('../Data/Vehicles_2015.csv')
    #accidents = pd.read_csv('Accidents_2015.csv')

def dqr(data):

    #%% Determinar el nombre de las variables
    columns = pd.DataFrame(list(data.columns.values), 
                           columns=['Nombres'],
                           index = list(data.columns.values))
    
    
    #%% Determinar el tipo de variable
    d_types = pd.DataFrame(data.dtypes,
                           columns=['Tipo'])
    
    #%% Determinar valores perdidos
    missing_values = pd.DataFrame(data.isnull().sum(axis=0),
                                  columns=['Valores perdidos'])
    
    #%% Determinar valores presentes
    present_values = pd.DataFrame(data.count(),
                                  columns=['Valores presentes'])
    
    #%% Determinar valores únicos
    unique_values = pd.DataFrame(data.nunique(),
                                  columns=['Valores unicos'])
    
    #%% Valores minimos y maximos de cada variable
    min_values = pd.DataFrame(columns=['Min'])
    max_values = pd.DataFrame(columns=['Max'])
    for col in list(data.columns.values):
        try:
            min_values.loc[col]=[data[col].min()]
            max_values.loc[col]=[data[col].max()]
        except: 
            pass
    
    #%% Juntar todas las tablas DATA QUALITY REPORT
    dqr = columns.join(d_types).join(missing_values).join(
            present_values).join(unique_values).join(min_values).join(max_values)
    return dqr

#%% Usar la funcion dqr()

reporte_accidents = dqr(accidents)
reporte_casualties = dqr(casualties)
reporte_vehicles = dqr(vehicles)