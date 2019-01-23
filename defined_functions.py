# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:45:08 2018

@author: Jishin
"""
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error


def get_score(prediction, lables): 
    '''
    Function to print R-squared and RMSE scores
    '''
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
    
#%%
def print_stats(df, col_name):
    if df[col_name].dtypes == "object":
        print("\n{} is a categorical field".format(col_name))
        print("\nNumber of missing values in {} is {}".format(col_name,df[col_name].isnull().sum()))
        print("\nPercentage of missing values in {} is {}%".format(col_name,round(100*df[col_name].isnull().sum()/len(df[col_name]),4)))
        print("\nPercentage distribution of the categories is\n{}".format(100*df[col_name].value_counts()/len(df[col_name])))
    else:
        print("\nNumber of missing values in {} is {}".format(col_name,df[col_name].isnull().sum()))
        print("\nPercentage of missing values in {} is {}%".format(col_name,round(100*df[col_name].isnull().sum()/len(df[col_name]),2)))
        print("\nThe mean of {} column is {}".format(col_name,df[col_name].mean()))
        print("\nThe median of {} column is {}\n".format(col_name,df[col_name].median()))
    
#%%
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    '''
    Function to predict the SalePrice by passing the classifier
    '''
    prediction_train = estimator.predict(x_trn)
    
    # Printing estimator
    print(estimator)
    
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)
#%%

