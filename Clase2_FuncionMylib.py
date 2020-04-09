#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:00:05 2019

@author: israelcastilloh
"""

### DATA QUALITY REPORT 

import pandas as pd
import numpy as np
from mylib import mylib


#%% Leer datos
accidents = pd.read_csv('../Data/Accidents_2015.csv')
casualties = pd.read_csv('../Data/Casualties_2015.csv')
vehicles = pd.read_csv('../Data/Vehicles_2015.csv')
    #accidents = pd.read_csv('Accidents_2015.csv')
    
 #%% Usar la funcion dqr()   
reporte_accidents = mylib.dqr(accidents)
reporte_casualties = mylib.dqr(casualties)
reporte_vehicles = mylib.dqr(vehicles)

#%% Consultas senciillas a un dataframe

Num_by_date = pd.DataFrame(pd.value_counts(accidents['Date']))
vehicles_day = pd.DataFrame(accidents.groupby(['Date'])['Number_of_Vehicles'].sum())
number_vehicles = pd.DataFrame(pd.value_counts(accidents['Number_of_Vehicles']))
casualties_day = pd.DataFrame(accidents.groupby(['Date'])['Number_of_Casualties'].sum())

vehicles_m = pd.DataFrame(pd.value_counts(vehicles['Sex_of_Driver']))

result = Num_by_date.join(vehicles_day).join(casualties_day)

vehicles_by_time = (accidents.groupby(['Time'])['Number_of_Vehicles'].sum())
accidents_by_time = pd.DataFrame(pd.value_counts(accidents['Time']))
accidents_by_day = pd.DataFrame(pd.value_counts(accidents['Day_of_Week']))
vehicles_age = pd.DataFrame(pd.value_counts(vehicles['Age_of_Driver']))
promedio = np.mean(vehicles_age.index[91:96])
#%% Realizar graficas



import matplotlib.pyplot as plt
plt.hist(accidents['Longitude'].dropna(), bins = 50,
         normed=True, cumulative = False, histtype='bar', 
         color = 'r')
plt.show()


import matplotlib.pyplot as plt
plt.hist(accidents['Latitude'].dropna(), bins = 50,
         normed=True, cumulative = False, histtype='bar', 
         color = 'r')
plt.show()

#%% Sobreponer figuras


import matplotlib.pyplot as plt
plt.hist(accidents['Day_of_Week'], bins = 7,
         normed=True, cumulative = True, 
         color = 'r', alpha = 0.2)

import matplotlib.pyplot as plt
plt.hist(accidents['Day_of_Week'], bins = 7,
         normed=True, cumulative = False, 
         color = 'b', alhpa = 0.5)

plt.show

#%% Histograma de forma cuantitativa
hist,bins=np.histogram(accidents['Day_of_Week'], 
                       bins = 7)

