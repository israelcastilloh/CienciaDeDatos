#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:59:17 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import pandas as pd


#%% Generar los datos
X = np.array([[2,3], [20,30], [-2,-3], [2,-3]])
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.axis('square')
plt.grid()
plt.show()


#%% Distancia euclidiana
D1 = sc.squareform(sc.pdist(X, 'euclidean'))

#%% Distancia euclidiana
D2 = 1-sc.squareform(sc.pdist(X, 'cosine')) #nos da el complemento 1-cos(theta)

#%% Dinstancia por indice de correlación
D3 = 1-sc.squareform(sc.pdist(X, 'correlation'))

