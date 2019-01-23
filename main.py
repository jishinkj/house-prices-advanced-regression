# Importing Libraries

import pandas as pd
#import numpy as np
import pickle

#%% Load the datasets

with open("data/column_dtype_dict.pickle", "rb") as f:
    column_dtype = pickle.load(f)

df = pd.read_csv("data/preprocessed.csv")  
#df = pd.read_csv("data/preprocessed.csv", dtype = column_dtype)
train = pd.read_csv("data/train.csv", usecols = ['Id','SalePrice'])

#%% Separate train and test sets

len_train = len(train['Id'])
test_ID = df.loc[len_train:,'Id']

# drop 'Id' from train and test sets
y = train['SalePrice']
df.drop(['Id'], axis = 1, inplace = True)

#%% One Hot Encode the categorical variables 

cat_columns = [column for column in df.columns if df[column].dtype == 'object']
    
df = pd.get_dummies(df, columns = cat_columns, drop_first = True)

#%% Separate the train and submission sets from the concatenated DF
train_df = df[:len_train]
test_df = df[len_train:]

#%% Train test split for modeling
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size = 0.2, random_state = 7)

#%% Modeling 
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

#%% 
y_predicted = rfr.predict(X_test)

#%% Evaluation Metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_predicted))

#%% Predict on submission dataset

submission = rfr.predict(test_df)

#%% EXPORT TO CSV

pd.DataFrame({'Id': test_ID, 'SalePrice': submission}).to_csv('v2.csv', index =False)


