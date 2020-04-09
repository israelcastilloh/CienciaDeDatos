#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:26:24 2019

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
from scipy.cluster import hierarchy
import scipy.spatial.distance as sc


#%% IMPORTAR DATOS
data = pd.read_csv('../Data/reporte-3.csv', encoding='latin1')

#%% APLICAR EL ESTUDIO DE CALIDAD DE DATOS
mireporte = mylib.dqr(data) 
#%%
data_nombre= pd.DataFrame(pd.value_counts(data['nombre']))
data_descripcion= pd.DataFrame(pd.value_counts(data['descripcion']))
data_periodicidad= pd.DataFrame(pd.value_counts(data['periodicidad']))
data_sentido= pd.DataFrame(pd.value_counts(data['sentido']))
data_unidad= pd.DataFrame(pd.value_counts(data['unidad_medida']))
data_fuente= pd.DataFrame(pd.value_counts(data['fuente']))


#%% limpiar
def remove_punctuation(x):
    try: 
        tmp = string.punctuation + '©' + '±' + 'º' + '' + '¡' + 'Ã'
        x = ''.join(ch for ch in x if ch not in tmp)
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
#%% LIMPIAR NOMBRES
data = pd.read_csv('../Data/reporte-3.csv', encoding='latin1')
data['nombre'] = data['nombre'].apply(remove_digits)
data['nombre'] = data['nombre'].apply(replace_text, args=('indÃ­gena','INDIGENA'))
data['nombre'] = data['nombre'].apply(replace_text, args=('MatrÃ­cula','MATRICULA'))
data['nombre'] = data['nombre'].apply(uppercase_text)
data['nombre'] = data['nombre'].apply(remove_punctuation)
data['nombre'] = data['nombre'].apply(replace_text, args=('TECNOLOG­A','TECNOLOGIA'))
data['nombre'] = data['nombre'].apply(replace_text, args=('Ã³','O'))
data['nombre'] = data['nombre'].apply(replace_text, args=('³','O'))
data['nombre'] = data['nombre'].apply(replace_text, args=('NMERO','NUMERO'))
data['nombre'] = data['nombre'].apply(replace_text, args=('CDULA','CEDULA'))
data['nombre'] = data['nombre'].apply(replace_text, args=('PBLICO','PUBLICO'))
data['nombre'] = data['nombre'].apply(replace_text, args=('MATEMTICAS','MATEMATICAS'))
data['nombre'] = data['nombre'].apply(replace_text, args=('MATEMTICA','MATEMATICAS'))
data['nombre'] = data['nombre'].apply(replace_text, args=('AOS','ADULTOS'))
data['nombre'] = data['nombre'].apply(replace_text, args=('NIAS','NIÑAS'))
data['nombre'] = data['nombre'].apply(replace_text, args=('NIOS','NIÑOS'))
data['nombre'] = data['nombre'].apply(replace_text, args=('ESPAOL','ESPAÑOL'))
data['nombre'] = data['nombre'].apply(replace_text, args=('REA','AREA'))
data['nombre'] = data['nombre'].apply(replace_text, args=('NDICE','INDICE'))
data['nombre'] = data['nombre'].apply(replace_text, args=('MATRIMONIÑOS','MATRIMONIOS'))
data_nombre= pd.DataFrame(pd.value_counts(data['nombre']))
#%% LIMPIAR OTRAS COLUMNAS
data['periodicidad'] = data['periodicidad'].apply(uppercase_text)
data['sentido'] = data['sentido'].apply(uppercase_text)
data_periodicidad= pd.DataFrame(pd.value_counts(data['periodicidad']))
data_sentido= pd.DataFrame(pd.value_counts(data['sentido']))
data['unidad_medida'] = data['unidad_medida'].apply(uppercase_text)
data['unidad_medida'] = data['unidad_medida'].apply(remove_punctuation)
data['unidad_medida'] = data['unidad_medida'].apply(replace_text, args=('CDULA','CEDULA'))
data['unidad_medida'] = data['unidad_medida'].apply(replace_text, args=('³','O'))
data_unidad= pd.DataFrame(pd.value_counts(data['unidad_medida']))

#%% RENOMBRAR LA COLUMNA DE VALOR ACTUAL COMO 2015, QUE ES LA ULTIMA TOMA DE LA BASE DE DATOS
data.rename(columns={'valor_actual':'2015'}, inplace = True)
mireporte = mylib.dqr(data)

#%% GENERAR UN RESUMEN CON LAS COLUMNAS IMPORTANTES
data_m = data[['nombre','sentido', 'unidad_medida', '2015', '2014', '2013', '2012', '2011', '2010']]
#%% SEPARAR HISTORICOS POR CADA INDICE
indices_hist = pd.DataFrame()
indices_hist['2010'] = data_m['2010'].values
indices_hist['2011'] = data_m['2011'].values
indices_hist['2012'] = data_m['2012'].values
indices_hist['2013'] = data_m['2013'].values
indices_hist['2014'] = data_m['2014'].values
indices_hist['2015'] = data_m['2015'].values
indices_hist.rename(index=data_m.nombre, inplace = True)
indices_hist = indices_hist.transpose()
indices_sim = indices_hist.transpose()

#%% INDICES ASCENDENTES Y DESCENDENTES
data_asc = data_m[['nombre', 'sentido']][data_m['sentido'] == "ASCENDENTE"]
data_des = data_m[['nombre', 'sentido']][data_m['sentido'] == "DESCENDENTE"]

#%%GRAFICAR INDICES DE ABANDONO ESCOLAR
indices_hist.iloc[:,0:4].plot.line( 
              figsize=(7, 7), legend=True, grid=True, marker='o')
plt.xticks(list(range(len(indices_hist))), indices_hist.index, fontsize=10)
plt.ylabel('PORCENTAJE')
plt.xlabel('Años')
plt.show()

#%%GRAFICAR INDICES DE EFICIENCIA TERMINAL POR NIVEL
indices_hist.iloc[:,22:25].plot.line( 
              figsize=(7, 7), legend=True, grid=True, marker='o')
plt.xticks(list(range(len(indices_hist))), indices_hist.index, fontsize=10)
plt.ylabel('PORCENTAJE')
plt.xlabel('Años')
plt.show()

#%%GRAFICAR INDICES DE # EGRESADOS POR NIVEL
indices_hist.iloc[:,25:28].plot.line( 
              figsize=(7, 7), legend=True, grid=True, marker='o')
plt.xticks(list(range(len(indices_hist))), indices_hist.index, fontsize=10)
plt.ylabel('# ALUMNOS')
plt.xlabel('Años')
plt.show()


#%%GRAFICAR INDICES DE REPROBACION
indices_hist.iloc[:,59:62].plot.line( 
              figsize=(7, 7), legend=True, grid=True, marker='o')
plt.xticks(list(range(len(indices_hist))), indices_hist.index, fontsize=10)
plt.ylabel('PORCENTAJE')
plt.xlabel('Años')
plt.show()
#%%GRAFICAR CADA INDICE INDIVIDUALMENTE
#for i in range(65):
    #indices_hist.iloc[:,i].plot.line( 
              #figsize=(6, 6), legend=False, grid=True, marker='o', title=data_m.iloc[i,0])
    #plt.xticks(list(range(len(indices_hist))), indices_hist.index, fontsize=10)
    #plt.ylabel(data_m.iloc[i,2])
    #plt.xlabel('Años')
    #plt.show()


#%%
Z = hierarchy.linkage(indices_sim, metric='euclidean', method = 'ward')
#%% Seleccionar elementos de los grupos formados
gruposmax = 30 # maximo numero de clusters, 
#en que elementos pertenece a que grupo
gruposfinal = pd.DataFrame(hierarchy.fcluster(Z, gruposmax, criterion = 'maxclust') )
gruposfinal.rename(index=data_m.nombre, inplace = True)


