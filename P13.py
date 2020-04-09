#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:21:08 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% IMPORTAR LOS DATOS (LEER LA IMAGEN)
img = mpimg.imread('../Data/images.png'x)
plt.imshow(img)

#%% REORDENAR LA IMAGEN EN UNA SOLA TABLA
d = img.shape
img_col = np.reshape(img, (d[0]*d[1], d[2]))

#%% CONVERTIR LOS DATOS A MEDIA CERO
media = img_col.mean(axis=0)
img_m = img_col - media

#%% OBTENER MATRIZ DE COVARIANZAS
img_cov = np.cov(img_m, rowvar=False)

#%% OBTENER LOS VALORES Y VECTORES PROPIOS
w,v = np.linalg.eig(img_cov)

#%% ANALIZAR LOS COMPONENTES PRINCIPALES
porcentaje = w/np.sum(w)
porcentaje_acum = np.cumsum(porcentaje)

#%% COMPRIMIR LA IMAGEN
componentes = w[0]
M_trans = np.reshape(v[:,0], (4,1))

img_new = np.matrix(img_m)*np.matrix(M_trans)

#%% RECUPERAR LA IMAGEN Y VISUALIZARLA
img_recuperada = np.matrix(img_new)*np.matrix(M_trans.transpose())
img_recuperada = img_recuperada + media

img_r = img.copy()
img_r[:,:, 0]= img_recuperada[:,0].reshape((d[0],d[1]))
img_r[:,:, 1]= img_recuperada[:,1].reshape((d[0],d[1]))
img_r[:,:, 2]= img_recuperada[:,2].reshape((d[0],d[1]))
img_r[:,:, 3]= img_recuperada[:,3].reshape((d[0],d[1]))

img_r[img_r<0]= 0

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_r)
