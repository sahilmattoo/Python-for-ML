# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 19:35:42 2021

@author: smattoo5
"""

# Read Data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # Not always recommended, but jsut so our notebook looks clean for this activity

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Splitting the data for training and testing out model
#from sklearn.base import TransformerMixin
from numpy import nan, isnan
from sklearn.impute import SimpleImputer
import joblib

scaler = StandardScaler()


import config as config
stratifyflag = config.straity_flag
splitsize = config.train_split_size

def readData():
    listofFiles = list()
    path = os.getcwd()
    path = path+"/Data"
    listofFiles = [each[2]for each in os.walk(path)][0]
    extension = [each.split(".")[1] for each in listofFiles]
    return listofFiles, extension


def loadData():
    df1 = pd.read_csv("Data/Flight data_Train.csv")  # Read the data regarding customer attributes
    df2 = pd.read_csv("Data/Surveydata_Train.csv")   # Feedback data from customers
    df = df2.set_index("Id").join(df1.set_index("ID"))
    
    return df

def findCols(df):
    
    category_cols = [c for c in df if df[c].dtype == np.dtype('O')]
    numeric_cols = [c for c in df if df[c].dtype != np.dtype('O')]
    
    return category_cols, numeric_cols

# def mapping(df):
#     df.replace({'extremely poor' : 0, 'poor' : 1, 'need improvement' : 2,
#                 'acceptable' : 3,'good' : 4, 'excellent' : 5, 'not_captured' : 2},
#                inplace = True)  

#     df.replace({'very inconvinient' : 0, 'Inconvinient' : 1, 'need improvement' : 2, 
#                 'manageable' : 3,'Convinient' : 4, 'very convinient' : 5}, 
#                inplace = True)
    
#     df.replace({'Loyal Customer' : 1, 'disloyal Customer' : 0,'Business travel' : 1, 
#                 'Personal Travel' : 0,'Female' : 0, 'Male' : 1,'satisfied' : 1, 
#                 'neutral or dissatisfied' : 0, 'Eco Plus': 0 , 'Eco': 1, 
#                 'Business': 2}, inplace = True)
    
#     return df

def mapValues(df):
    map1 = config.map1
    map2 = config.map2
    map3 = config.map3
    df.replace(map1, inplace = True)
    df.replace(map2, inplace = True)
    df.replace(map3, inplace = True)
    
    return df

def imputeValues (df, c, n):
    info = df.isna().sum()
    
    imputeFeatures = [info.index[each] for each in range(len(info)) 
                      if info[each] > 0]
    
    return imputeFeatures    

   
    
def defineX_y(df):
    X = df.drop(config.drop_x, axis =1)
    y = df[config.col_y]
    return X,y

def splitData(X,y, stratifiedData, size):
    if stratifiedData:
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1, 
                                                            stratify = y, 
                                                            train_size= size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1, 
                                                            train_size= size)

    print(X_train.shape)
    print(y_train.shape)
    return X_train, X_test, y_train, y_test


def dataprepration():
    
    
    df = loadData() 
    c, n =  findCols(df)
    
    #df = mapping(df)
    df = mapValues(df)
    f = imputeValues (df, c, n)
    #online_df = df.loc[:, ['Ease_of_Onlinebooking', 'Online_boarding', 'Online_support']]
    #online_df['avg_feedback_of_online_services'] = online_df.mean(axis = 1)
    #scaled_values = scaler.fit(df[scale])
    X, y  = defineX_y(df)
       
    X_train, X_test, y_train, y_test = splitData(X,y, stratifiedData = stratifyflag, size = splitsize)

    joblib.dump(X_train, 'Data/X_train')
    joblib.dump(X_test, 'Data/X_test')
    joblib.dump(y_train, 'Data/y_train')
    joblib.dump(y_test, 'Data/y_test')

    return X_train, X_test, y_train, y_test


def simpleImputeValue(trainingdata, strategy):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputeX = imp_mean.fit(trainingdata)
    return imputeX
    

def transformations():
    X_train, X_test, y_train, y_test = dataprepration()
    impX = simpleImputeValue(X_train, strategy = config.strategy)
    scl = scaler.fit(X_train)
    joblib.dump(impX, 'Models/imputeMethod')
    joblib.dump(scl, 'Models/scalingMethod' )
    
    return impX,  scl
    
if __name__ == '__main__':
    transformations()
    print("Preprocessing in progress")
 
