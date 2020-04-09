#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:37:47 2019

@author: israelcastilloh
""""
###################################### EJ 3 ###################################
######################################  RL  ###################################

#%%
data3 = pd.read_csv('../Data/dataset_4_b.csv',header=None)

#%% Separar datos de entranamiento y prueba
X=data3.iloc[:,:-1]
Y=data3.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)    


#%%
ngrado = 2
poly = PolynomialFeatures(ngrado)
Xa_train = poly.fit_transform(X_train)
Xa_test = poly.fit_transform(X_test)
modelo = linear_model.LogisticRegression()
modelo.fit(Xa_train,Y_train)
Yhat_test = modelo.predict(Xa_test)


#%%
dummies1 = pd.get_dummies(Y_test)
dummies2 = pd.get_dummies(Yhat_test)

y1=dummies1[0]
y2=dummies1[1]
y3=dummies1[2]
Yhat1= dummies2[0]
Yhat2= dummies2[1]
Yhat3= dummies2[2]
accuracy =(np.mean([accuracy_score(y1,Yhat1),accuracy_score(y2,Yhat2),accuracy_score(y3,Yhat3)]))
precision=(np.mean([precision_score(y1,Yhat1),precision_score(y2,Yhat2),precision_score(y3,Yhat3)]))
recall= (np.mean([recall_score(y1,Yhat1),recall_score(y2,Yhat2),recall_score(y3,Yhat3)]))

plt.bar(['Accu','Prec','Rec'],[accuracy,precision,recall])
print(accuracy,precision,recall)

#%%





###################################### SVM ####################################
#%%
x=data3.iloc[:,:-1]
y=data3.iloc[:,-1]
#%% Crear el modelo SVC. Clasificador de vector soporte. 
modelo = svm.SVC(kernel= 'rbf')
modelo.fit(x,y)
Yhat=modelo.predict(x) # Aquí comparamos los datos con el accuracy, recall etc

#%% Medir precisión del modelo clasificador mediante dummies.
dummies1 = pd.get_dummies(y)
dummies2 = pd.get_dummies(Yhat)
# Se va a medir la precisión columna a columna y se hara un promedio de las 3 precisiones
y1=dummies1[0]
y2=dummies1[1]
y3=dummies1[2]
Yhat1= dummies2[0]
Yhat2= dummies2[1]
Yhat3= dummies2[2]
accuracy =(np.mean([accuracy_score(y1,Yhat1),accuracy_score(y2,Yhat2),accuracy_score(y3,Yhat3)]))
precision=(np.mean([precision_score(y1,Yhat1),precision_score(y2,Yhat2),precision_score(y3,Yhat3)]))
recall= (np.mean([recall_score(y1,Yhat1),recall_score(y2,Yhat2),recall_score(y3,Yhat3)]))

plt.bar(['Accu','Prec','Rec'],[accuracy,precision,recall])
print(accuracy,precision,recall)


#%%
plt.scatter(x.iloc[:,0],x.iloc[:,1],c=Yhat)
plt.show()
plt.scatter(x.iloc[:,0],x.iloc[:,2],c=Yhat)
plt.show()
plt.scatter(x.iloc[:,1],x.iloc[:,2],c=Yhat)
plt.show()







