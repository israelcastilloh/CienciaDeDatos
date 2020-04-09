#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:36:12 2019

@author: israelcastilloh
"""

#Limpieza de bases de datos y manejo de texto
import pandas as pd
import string 
from datetime import datetime
from mylib import mylib

#%% Importar la tabla o base de datos
dirty = pd.read_csv('../Data/dirty_data_v3.csv', 
                    encoding='latin-1')

#%% Funcion para retirar signos de puntuacion
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

#%% Aplicar las funciones
dirty['apellido'] = dirty['apellido'].apply(lowercase_text)
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('0','o'))
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('-',' '))
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('_',' '))
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('4','a'))
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('1','i'))
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('2','z'))
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('8','b'))
dirty['apellido'] = dirty['apellido'].apply(replace_text, args=('ochoanavarro','ochoa navarro'))
dirty['apellido'] = dirty['apellido'].apply(remove_digits)
dirty['apellido'] = dirty['apellido'].apply(remove_punctuation)

#%% Limpiar estado civil
dirty['estado civil'] = dirty['estado civil'].apply(lowercase_text)

#%% Limpiar lugar de nacimiento
dirty['lugar de nacimiento'] = dirty['lugar de nacimiento'].apply(lowercase_text)

















