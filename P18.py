#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:15:15 2019

@author: israelcastilloh
"""

##SUPPORT VECTOR MACHINE

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#%%
np.random.seed(5)
X = np.r_[np.random.randn(20,2)-[2,2],
          np.random.randn(20,2)+[2,2]]
Y = [0]*20+[1]*20

plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()

#%%
modelo = svm.SVC(kernel = 'linear')
#modelo = svm.SVC(kernel = 'poly', degree = 2)
#modelo = svm.SVC(kernel = 'rbf')
modelo.fit(X,Y)

Yhat = modelo.predict(X)

#%% Dibujar el plano de separacion
W = modelo.coef_[0] 
m = -W[0]/W[1]
xx = np.linspace(-5,5)
yy = m*xx-(modelo.intercept_[0]/W[1])

VS = modelo.support_vectors_

plt.plot(xx,yy,'k-')
plt.scatter(X[:,0],X[:,1], c=Y)
plt.scatter(VS[:,0],VS[:,1], s=80, facecolors='r')
plt.show()