# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:42:28 2021

@author: smattoo5
"""

import preprocessing as pre
import joblib

### Load the data 
X_train = joblib.load('Data/X_train')
X_test = joblib.load('Data/X_test')
y_train = joblib.load('Data/y_train')
y_test = joblib.load('Data/y_test')
imp = joblib.load('Models/imputeMethod')
scl = joblib.load('Models/scalingMethod' )
    

### Impute and scale Data
def pipeline(X):
    imputeX = imp.transform(X)
    scaleX = scl.transform(imputeX)
    print(scaleX.shape)
    return scaleX

#### Create a Model
data = pipeline(X = X_train)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(data, y_train)
score4 = rf.score(data, y_train)
print(f'Random Forest accuracy score = {score4}')
joblib.dump(rf, 'Models/randomforest')

