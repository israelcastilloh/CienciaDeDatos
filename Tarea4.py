
"""
Created on Mon Mar 11 10:12:29 2019

@author: israelcastilloh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from scipy.cluster import hierarchy
import scipy.spatial.distance as sc

#%% Seleccionar las columnas con calificaciones
data = pd.read_excel('../Data/Test de películas(1-16).xlsx', 
                     encoding = 'latin_1') #para leer sin acento
csel = np.arange(6, 243, 3)
cnames = list(data.columns.values[csel])
datan =data[cnames]

#%%Aplicar algoritmo de clustering
#Grafica de codo

inercias = np.zeros(15) #preallocation
for k in np.arange(len(inercias))+1:
    model = KMeans(n_clusters=k, init='random')
    model = model.fit(datan)
    inercias[k-1]= model.inertia_
    
plt.plot(np.arange(len(inercias))+1, inercias)
plt.xlabel('Num de grupos')
plt.ylabel('Inercia Total')
plt.show()

#%% Clasificar los datos segun las graficas de codo
model = KMeans(n_clusters=4, init='random')
model = model.fit(datan)
grupos1 = model.predict(datan)

 #%%Graficar los centroides con KMEANS
centroides = model.cluster_centers_
plt.plot(centroides.transpose())
plt.grid()
plt.show()




#%% Aplicar el algoritmo de clustering con Hierarchy
Z = hierarchy.linkage(datan, metric='euclidean', method = 'complete')

plt.figure(figsize = (7,4))
plt.title('dendrograma completo')
plt.xlabel('Indice de la muestra')
plt.ylabel('Distancia o Similitud')
dn = hierarchy.dendrogram(Z)
plt.show()


#%% Criterio de codo (Parte 1)
last = Z[-15:,2]
last_rev = last[::-1]
idxs = np.arange(1,len(last_rev)+1)
plt.plot(idxs, last_rev)
plt.xlabel('# grupos')
plt.ylabel('distancia equivalente')
plt.grid()
plt.show()

#%% Seleccionar elementos de los grupos formados
distmax = 4
grupos = hierarchy.fcluster(Z, distmax, criterion = 'maxclust') #en que elementos pertenece a que grupo

