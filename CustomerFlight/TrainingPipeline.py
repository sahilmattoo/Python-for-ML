# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 22:01:16 2021

@author: smattoo5
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
#from category_encoders import OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder


import config as config
import preprocessing as pre 


numeric_features = config.numeric_features
feedback_features = config.feedback_features
other_cat_cols = config.other_cat_cols

#TRANSFORMERS



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


feedback_feature_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='not_captured')),
    ('label_encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())])


other_cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='not_captured')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('feed_col', feedback_feature_transformer, feedback_features),
        ('other_cat_col', other_cat_transformer, other_cat_cols )
    ])




from sklearn.ensemble import RandomForestClassifier
#Adding into Pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(bootstrap= True,max_depth= 30,
                                                            max_features= 'auto',min_samples_leaf= 1,
                                                            n_estimators= 100))])


data  = pre.loadData()
#Getting X and y
X1 = data.drop(['Satisfaction', 'ArrivalDelayin_Mins'], axis = 1)
y1 = pd.get_dummies(data['Satisfaction'])
#Data SPlit
from sklearn.model_selection import train_test_split # Splitting the data for training and testing out model
X_trains, X_tests, y_trains, y_tests = train_test_split(X1,y1, random_state = 1, stratify = y1)
#Fitting Pipeline 
clf.fit(X_trains, y_trains)


