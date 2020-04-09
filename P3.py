#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:05:26 2019

@author: israelcastilloh
"""

import pandas as pd
import numpy as np
import sklearn.metrics as skm
import scipy.spatial.distance as sc

#%% Leer los datos de la encuesta

data = pd.read_excel('../Data/Test de películas(1-16).xlsx', 
                     encoding = 'latin_1') #para leer sin acentos
                     
#%% Seleccionar las columnas con calificaciones

csel = np.arange(6, 243, 3)
cnames = list(data.columns.values[csel])
datan =data[cnames]

#%% Valoración promedio por pelicula
movie_prom = datan.mean(axis=0) #por pelicula calif
user_prom = datan.mean(axis=1) #por usuario calif

#%% Convertir calificaciones a binarios
for col in cnames:
    indx = datan[col]<3
    datan[col][indx]=0
    #indx = datan[col]>=3
    #datan[col][indx]=1
    datan[col][datan[col]>=3] = 1
    #datan[col][datan[col]<3] = 1
    
#%% Calcular las distancias 
D1 = sc.squareform(sc.pdist(datan, 'jaccard'))
Isim1 = 1-D1 

#%% Seleccionar usuario para recomendación de otros parecidos
user = 7
Isim_user = Isim1[user]
Isim_user_sort = np.sort(Isim_user)
indx_user = np.argsort(Isim_user)

#%% Recomendación de peliculas versión 1 (buscar el usuario más parecido)
USER = datan.loc[user]
USER_sim = datan.loc[indx_user[-2]]

indx_recomend1 = (USER_sim==1)&(USER==0)
recomend1 = list(USER.index[indx_recomend1])

#%% Recomendación de peliculas version 2
#Buscar a los n usuarios mas parecidos
USER = datan.loc[user]
USER_sim = np.mean(datan.loc[indx_user[-6:-1]], axis=0)
USER_sim[USER_sim<=0.5] = 0
USER_sim[USER_sim>0.5] = 1


indx_recomend2 = (USER_sim==1)&(USER==0)
recomend2 = list(USER.index[indx_recomend2])








