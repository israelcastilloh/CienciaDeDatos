#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:48:47 2019

@author: israelcastilloh
"""


import numpy as np

array = []
array3=[]

#LONGITUD DE LA SERIE DE TIEMPO
p = 100
#LONGITUD DE MINISERIES
n = 6


for i in range (1,p+1):
    #array = np.append(array, i) #para resultado original
    #x = np.random.randint(200) #para var. aleatorias de 0 a 200
    z = np.random.normal(0, 1) #para var. aleatorias con dist. normal estándar.
    array = np.append(array, z) 
    
j = 0
while j < (p-n+1):
    #x = array[j:j+6]
    #print(x)
    array3=np.concatenate((array3, array[j:j+n]))
    j+=1
finaldestiny = np.reshape(array3, (p-n+1, n))
