#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:08:05 2019

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


#%% IMPORTAR DATOS
suelo = pd.read_csv('../Data/cancelacion_2017.csv', encoding='latin1')

#%% APLICAR EL ESTUDIO DE CALIDAD DE DATOS
mireporte = mylib.dqr(suelo) 

#%% limpiar
def remove_punctuation(x):
    try: 
        x = ''.join(ch for ch in x if ch not in string.punctuation)
    except:
        pass
    return x

#tmp = string.punctuation + '`' y se pone, not in tmp
#%% Remover digitos
def remove_digits(x):
    try:
        x = ''.join(ch for ch in x if ch not in string.digits)
    except:
        pass
    return x

#%% Remover espacios en blanco
def remove_whitespaces(x):
    try:
        x = ''.join(x.split())
    except:
        pass
    return x

#%% Remplazar texto
def replace_text(x, to_replace, replacement):
    try:
        x = x.replace(to_replace, replacement)
    except:
        pass
    return x

#%% Convertir a mayusculas
def uppercase_text(x):
    try:
        x = x.upper()
    except:
        pass
    return x

#%% Convertir a minusculas
def lowercase_text(x):
    try:
        x = x.lower()
    except:
        pass
    return x

#%% Dejar solo digitos
def only_digits(x):
    try:
        x = ''.join(ch for ch in x if ch in string.digits)
    except:
        pass
    return x



#%%Limpiar datos
 
suelo = pd.read_csv('../Data/cancelacion_2017.csv', encoding='latin1')

suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('.','w'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(lowercase_text)
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(remove_punctuation)
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(remove_digits)
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(remove_whitespaces)
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('urbano.','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('urbana','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('urnamo','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('urabana','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('urabano','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('hurbano','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('urbabno','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('noconsta','inespecifico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('noespecifica','inespecifico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('cesionparadestinos','cesion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('rusticoq','rustico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('rústico','rustico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('turisticohotelero','turistico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('turisticohabitacion','turistico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habtacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habiatcional', 'habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacinal','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitcional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('abitacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habiatacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habtiacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habiacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacioal','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('hatiacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habatacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitaconal','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habiotacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('haitacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacionasl','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitaiconal','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitaciona','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacionmal','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacioanal','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacionla','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habaitacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habiracional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('has','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('ha','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacional','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacionbitacion','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('m','mixto'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('comixtoercial','comercial'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('estacionamixtoiento','estacionamiento'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('mixtoixto','mixto'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('comixtoun','comun'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('habitacionw','habitacion'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('urbanow','urbano'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('inespecificow','inespecifico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('wmixtow','mixto'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(replace_text, args=('w','inespecifico'))
suelo['Uso Suelo'] = suelo['Uso Suelo'].apply(uppercase_text)
#%% APLICAR EL ESTUDIO DE CALIDAD DE DATOS
suelo_m= pd.DataFrame(pd.value_counts(suelo['Uso Suelo']))
mireporte_m = mylib.dqr(suelo) 
#%%

#%% Graficos
suelo_m.plot()
plt.xlabel('Suelos ')
plt.ylabel('Cantidades')

plt.show()







