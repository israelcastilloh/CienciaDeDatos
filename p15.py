#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:31:30 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import scipy.optimize as opt

#%% Generar los datos deudores y pagadores
X, Y  = make_blobs(n_samples=100, centers=[[0,0],[5,5]],
                   cluster_std=0.5,
                   n_features=2)

plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#%% Funcion logistica
def fun_log(V):
    return 1/(1+np.exp(-V))

#%% Regresion logistica
    def reg_log(W,X,Y):
        V = np.matrix(X)*np.matrix(W).transpose()
        return np.array(fun_log(V)[:,0])

#%% Funcion de costo
        def fun_cost(W,X,Y):
            Yhat = reg_log(W,X,Y)
            J = np.sum(-Y*np.log(Yhat)-(1-Y)*np.log(1-Yhat))/len(Y)
            return J
        
#%% Inicializar las variables de optimizacion
Xa = np.append(np.ones((len(Y),1)),X,axis=1)
m, n = np.shape(Xa)
W = np.zeros(n)

#optimizacion
res = opt.minimize(fun_cost,W, args=(Xa,Y))
W = res.x

#%% Simular mi modelo
Yhat = np.round(reg_log(W,Xa,Y))