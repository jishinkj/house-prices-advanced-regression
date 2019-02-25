#CODE FROM - https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
#from sklearn.metrics import r2_score, mean_squared_error

from defined_functions import *
import warnings
warnings.filterwarnings('ignore')

#%%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#%%

train_sum_NA = train.isnull().sum()
test_sum_NA = test.isnull().sum()
train_percent_NA = train.isnull().sum()/len(train.SalePrice)
test_percent_NA = test.isnull().sum()/len(test.Id)

NAs = pd.concat([train_sum_NA,train_percent_NA,test_sum_NA,test_percent_NA], 
                axis=1, keys=['Train', '% Missing','Test', '% Missing'])

NAs[NAs.sum(axis=1) > 0]

#%%
#train_labels = train.pop('SalePrice')

#%% DROP THE SalePrice COLUMN AND JOIN THE TRAIN AND TEST CSVs 
SalePrice_col = train.SalePrice
train_IDVs = train.iloc[:,:-1]

features = pd.concat([train_IDVs, test], keys=['train', 'test'])

#%% SAVE THE CONCATANATED DATASET TO A CSV
#features.to_csv('concatedDatasetBeforeCleaning.csv',index = False)

# LOAD THIS DATASET
features = pd.read_csv('data/concatedDatasetBeforeCleaning.csv')

#%% IGNORE
'''
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
               'BsmtUnfSF', 'Heating', 'LowQualFinSF','BsmtFullBath', 'BsmtHalfBath', 
               'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
               'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],axis=1, inplace=True)
'''
# =============================================================================
# Missing Value Imputation and Type Conversion
# =============================================================================

'''
columns = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
'''
# fruit = pd.Series(['banana'] * 57 + ['apple'] * 54 + [None] * 10, name='fruit')


# nullfruit = fruit.isnull()
# fruit.loc[nullfruit] = fruit.dropna().sample(nullfruit.sum()).values
#%% Id
df = pd.DataFrame()
df['Id'] = features['Id']

# MSSubClass - Identifies the type of dwelling involved in the sale.
# no missing values, convert to str type
df['MSSubClass'] = features['MSSubClass'].astype(str)
print_stats(df, 'MSSubClass')

# MSZoning - Identifies the general zoning classification of the sale.

# categorical variable, 4 NAs in test set
print_stats(features, 'MSZoning')

# 4 missing values, 77% values belong to RL class. Impute with RL (mode)

df['MSZoning'] = features['MSZoning'].astype(str)
df['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

#  LotFrontage - Linear feet of street connected to property
print_stats(features, 'LotFrontage')

# Percentage missing = 16.65% , Mean = 69.305 , Median = 68 , Imputed with median  
df['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].median())

# LotArea - Lot size in square feet
# No missing values
print_stats(features, 'LotArea')

# Percentage missing =0 , Mean =10168.11 , Median =9453  
df['LotArea'] = features['LotArea']

# Street - Type of road access to property
# no misssing values (99.5% values are Pave, 0.411% values Grvl)
print_stats(features, 'Street')

df['Street'] = features['Street']

# Alley - Type of alley access to property
print_stats(features, 'Alley')

# Percentage missing = 93.21% , Classes are PAve and Grvl. NA means no Alley. Impute with NoAlley
df['Alley'] = features['Alley'].fillna("NoAlley")

# LotShape - General shape of property
print_stats(features, 'LotShape')

# cat variable, No missing values
# Classes - Reg, IR!, IR2, IR3
df['LotShape'] = features['LotShape']

# LandContour - Flatness of the property
print_stats(features, 'LandContour')

# No missing values - Classes Lvl, HLS, Bnk, Low
df['LandContour'] = features['LandContour']

# Utilities - Type of utilities available
print_stats(features, 'Utilities')

# cat variable, 2 missing values
# Percentage missing =0.0007 , Classes - AllPub 99.89%, NoSeWa 0.03%, impute with mode
df['Utilities'] = features['Utilities'].fillna(features['Utilities'].mode()[0])

# LotConfig - Lot configuration
# cat variable with 5 classes, no missing values
print_stats(features, 'LotConfig')

# No missing values
df['LotConfig'] = features['LotConfig']

# LandSlope - Slope of property
# categorical variable with 3 classes
print_stats(features, 'LandSlope')
# No missing values
df['LandSlope'] = features['LandSlope']

# Neighborhood - Physical locations within Ames city limits

print_stats(features, 'Neighborhood')

# No missing values, has 27 classes in this column
df['Neighborhood'] = features['Neighborhood']

# Condition1 - Proximity to various conditions

print_stats(features, 'Condition1')
# no missing values, has 9 classes 
df['Condition1'] = features['Condition1']

# Condition2 - Proximity to various conditions (if more than one is present)

print_stats(features, 'Condition2')
# no missing values, has 8 classes 
df['Condition2'] = features['Condition2']

# BldgType - Type of dwelling

print_stats(features, 'BldgType')
# no missing values, 5 classes
df['BldgType'] = features['BldgType']

# HouseStyle - Style of dwelling

print_stats(features, 'HouseStyle')

# no missing values, 8 classes
df['HouseStyle'] = features['HouseStyle']

# OverallQual - Rates the overall material and finish of the house

print_stats(features, 'OverallQual')
# Should it be converted to categorical type??? 
# no missign values
df['OverallQual'] = features['OverallQual']

# OverallCond - Rates the overall condition of the house
print_stats(features, 'OverallCond')
# Should it be converted to categorical type??? 
# If kept as numeric, check the correlation between OverallQual and OverallCond
df['OverallCond'] = features['OverallCond']
# YearBuilt - Original construction date
features['YearBuilt'] = features['YearBuilt'].astype(str)
print_stats(features, 'YearBuilt')

# No missing values
df['YearBuilt'] = features['YearBuilt']

# YearRemodAdd
features['YearRemodAdd'] = features['YearRemodAdd'].astype(str)
print_stats(features, 'YearRemodAdd')

# No missing values
df['YearRemodAdd'] = features['YearRemodAdd']

# RoofStyle - Type of roof
print_stats(features, 'RoofStyle')

# no missing values
df['RoofStyle'] = features['RoofStyle']

# RoofMatl - Roof Material
print_stats(features, 'RoofMatl')
# no missing values

df['RoofMatl'] = features['RoofMatl']

# Exterior1st - Exterior covering on house

print_stats(features, 'Exterior1st')

# 1 missing value, 17 classes, "VinylSd" has 35% values. Impute with that. 
df['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])


# Exterior2nd - Exterior covering on house (if more than 1 material)

print_stats(features, 'Exterior2nd')
df['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
# 1 missing value, "VinylSd" has 34% values. Impute with that.

# MasVnrType - Masonry veneer type
print_stats(features, 'MasVnrType')
# 24 missing values, None means No Masonry Veneer
# Percentage missing = 0.82%, None has 59% values, Impute with "None"
df['MasVnrType'] = features['MasVnrType'].fillna("None")

# MasVnrArea - Masonry veneer area in square feet
print_stats(features, 'MasVnrArea')

# 23 missing values, Percentage missing = 0.01, Mean =102.2013 , Median = 0, 0 means no masonry veneer  
df['MasVnrArea'] = features['MasVnrArea'].fillna(0)

# ExterQual - Evaluates the quality of the material on the exterior

print_stats(features, 'ExterQual')

# no missing values, 5 classes
df['ExterQual'] = features['ExterQual']

# ExterCond - Evaluates the present condition of the material on the exterior
print_stats(features, 'ExterCond')

# no missing values, 5 classes, 86% in 'TA'  
df['ExterCond'] = features['ExterCond']

# Foundation - Type of foundation
print_stats(features, 'Foundation')

# no missing values
df['Foundation'] = features['Foundation']

# BsmtQual - Evaluates the height of the basement
print_stats(features, 'BsmtQual')

# 81 missing values, Percentage missing =2.77%, Even ditributed classes, Impute with NoBasement

#nullBsmtQual = features['BsmtQual'].isnull()
#features['BsmtQual'].loc[nullBsmtQual] = features['BsmtQual'].dropna().sample(nullBsmtQual.sum()).values
#df['BsmtQual'] = features['BsmtQual']

df['BsmtQual'] = features['BsmtQual'].fillna("NoBasement")

# BsmtCond - Evaluates the general condition of the basement
print_stats(features, 'BsmtCond')

# 82 missing values, 4 classes. Missing value means NoBasement
df['BsmtCond'] = features['BsmtCond'].fillna("NoBasement")

# BsmtExposure - Refers to walkout or garden level walls
print_stats(features, 'BsmtExposure')

# 82 missing values, 4 classes, 65% and 14%, do percentage imputation
df['BsmtExposure'] = features['BsmtExposure'].fillna("NoBasement")

# BsmtFinType1 - Rating of basement finished area
print_stats(features, 'BsmtFinType1')
# 79  missing values - percentage imputation

df['BsmtFinType1'] = features['BsmtFinType1'].fillna("NoBasement")

# BsmtFinSF1 - Type 1 finished square feet
print_stats(features, 'BsmtFinSF1')
# 1 missing value, Impute with 0
# Percentage missing = 0, Mean =441.4232 , Median =368.5 

df['BsmtFinSF1'] = features['BsmtFinSF1'].fillna(features['BsmtFinSF1'].median())

# BsmtFinType2 - Rating of basement finished area (if multiple )
print_stats(features, 'BsmtFinType2')

# 80 missing values, 6 classes, impute with "NoBasement"
df['BsmtFinType2'] = features['BsmtFinType2'].fillna("NoBasement")

# BsmtFinSF2 - Type 2 finished square feet
print_stats(features, 'BsmtFinSF2')

# 1 missing value, Mean =49.58 , Median =0 , Impute with 0 
df['BsmtFinSF2'] = features['BsmtFinSF2'].fillna(features['BsmtFinSF2'].median())

# BsmtUnfSF - Unfinished square feet of basement area
print_stats(features, 'BsmtUnfSF')

# 1 missing value, Mean =560.772 , Median =467 , Imputed with median

df['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(features['BsmtUnfSF'].median())

# TotalBsmtSF - Total square feet of basement area
print_stats(features, 'TotalBsmtSF')

# 1 missing value, Mean = 1051.777, Median =989.5 , Imputed with median

df['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(features['TotalBsmtSF'].median())
# Heating - Type of heating
print_stats(features, 'Heating')

# no missing values, 6 classes, 'GasA' has 98% values
df['Heating'] = features['Heating']

# HeatingQC - Heating quality and condition
print_stats(features, 'HeatingQC')

# no missing values, 5 classes,

df['HeatingQC'] = features['HeatingQC']

# CentralAir - Central air conditioning
print_stats(features, 'CentralAir')

# 0 missing values, 2 classes, Y and N, 6% values in N
df['CentralAir'] = features['CentralAir']

# Electrical - Electrical System
print_stats(features, 'Electrical')

# 1 missing value - 5 classes, 91% values in "SBrkr", Impute with mode
df['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

# 1stFlrSF - First floor square feet
print_stats(features, '1stFlrSF')

# No missing values
df['1stFlrSF'] = features['1stFlrSF']

# 2ndFlrSF
print_stats(features, '2ndFlrSF')

# No missing values
df['2ndFlrSF'] = features['2ndFlrSF']

# LowQualFinSF - Low quality finished square feet (all floors)
print_stats(features, 'LowQualFinSF')

# No missing values
df['LowQualFinSF'] = features['LowQualFinSF']

# GrLivArea - Above grade (ground) living area square feet
print_stats(features, 'GrLivArea')

# No missing values
df['GrLivArea'] = features['GrLivArea']
# BsmtFullBath - basement full bathrooms
print_stats(features, 'BsmtFullBath')

# 2 missing values, Number of bathrooms - 0, 1, 2, 3 and nan, mode is 0, impute with mode
df['BsmtFullBath'] = features['BsmtFullBath'].fillna(features['BsmtFullBath'].mode()[0])

# BsmtHalfBath
print_stats(features, 'BsmtHalfBath')

# 2 missing values, Number of bathrooms - 0, 1, 2, 3 and nan, mode is 0, impute with mode 
df['BsmtHalfBath'] = features['BsmtHalfBath']

# FullBath
print_stats(features, 'FullBath')

# No missing values 
df['FullBath'] = features['FullBath']

# HalfBath
print_stats(features, 'HalfBath')

# No missing values 
df['HalfBath'] = features['HalfBath']

# Bedroom - Bedrooms above grade
print_stats(features, 'BedroomAbvGr')

# No missing values 
df['BedroomAbvGr'] = features['BedroomAbvGr']

# Kitchen - Kitchens above grade
print_stats(features, 'KitchenAbvGr')

# No missing values 
df['KitchenAbvGr'] = features['KitchenAbvGr']

# KitchenQual - 
print_stats(features, 'KitchenQual')

# 1 missing value, 4 classes, 51% in "TA", Imputed with mode
df['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

# TotRmsAbvGrd - Total rooms above grade (does not include bathrooms)
print_stats(features, 'TotRmsAbvGrd')

# No missing values 
df['TotRmsAbvGrd'] = features['TotRmsAbvGrd']

# Functional - Home functionality (Assume typical unless deductions are warranted)
print_stats(features, 'Functional')

# 2 missing values, 8 classes, 93% in "Typ", Imputed with Typ
df['Functional'] = features['Functional'].fillna(features['Functional'].mode()[0])

# Fireplaces - Number of fireplaces
print_stats(features, 'Fireplaces')

# 0 missing values
df['Fireplaces'] = features['Fireplaces']

# FireplaceQu - Fireplaces
print_stats(features, 'FireplaceQu')

# Percentage missing = 48.65%, Missing values means no fireplace, Imputed with "NoFireplace"
df['FireplaceQu'] = features['FireplaceQu'].fillna("NoFireplace")

# GarageType - Garage location
print_stats(features, 'GarageType')

# 157 missing values, missing value means NoGarage, impute with "NoGarage"
df['GarageType'] = features['GarageType'].fillna("NoGarage")

# GarageYrBlt - Year garage was built
print_stats(features, 'GarageYrBlt')
# 159 missing values

features['GarageYrBlt'] = features['GarageYrBlt'].astype(str)

df['GarageYrBlt'] = features['GarageYrBlt'].fillna("NoGarage")

# GarageFinish - Interior finish of the garage
print_stats(features, 'GarageFinish')

# 159 missing values, impute with "NoGarage"
df['GarageFinish'] = features['GarageFinish'].fillna("NoGarage")

# GarageCars - Size of garage in car capacity
print_stats(features, 'GarageCars')

# 1 missing value, Mode is 2, Impute with 2, Keep it as numeric field itself 
df['GarageCars'] = features['GarageCars'].fillna(features['GarageCars'].mode()[0])

# GarageArea - Size of garage in square feet
print_stats(features, 'GarageArea')

# 1 missing value, lot of zero values, Zero means NoGarage, Mean =472.874 , Median =480, Impute with median  
df['GarageArea'] = features['GarageArea'].fillna(features['GarageArea'].median())

# GarageQual - Garage quality
print_stats(features, 'GarageQual')

# 159 missing values, Percentage missing = 5.45%, Imputed with NoGarage
df['GarageQual'] = features['GarageQual'].fillna("NoGarage")

# GarageCond - Garage condition
print_stats(features, 'GarageCond')

# 159 missing values, Percentage missing = 5.45%, Imputed with NoGarage
df['GarageCond'] = features['GarageCond'].fillna("NoGarage")

# PavedDrive - Paved Driveway
print_stats(features, 'PavedDrive')
#       Y	Paved 
#       P	Partial Pavement
#       N	Dirt/Gravel
# no missing values, 90% values in "Y"

df['PavedDrive'] = features['PavedDrive']

# WoodDeckSF - Wood Deck Area in square feet
print_stats(features, 'WoodDeckSF')
# no missing values, Mean =93.7098 , Median =0 , 0 value means Wood Deck
df['WoodDeckSF'] = features['WoodDeckSF']

# OpenPorchSF - Screen porch area in square feet
print_stats(features, 'OpenPorchSF')

# no missing values, Mean =47.48 , Median =26 , 0 value means NoPorch
df['OpenPorchSF'] = features['OpenPorchSF']

# EnclosedPorch - Enclosed porch area in square feet
print_stats(features, 'EnclosedPorch')

# no missing values, Mean =23.098 , Median =0 , 0 value means NoPorch
df['EnclosedPorch'] = features['EnclosedPorch']

# 3SsnPorch - Three season porch area in square feet
print_stats(features, '3SsnPorch')

# Percentage missing = 0, Mean =2.6022 , Median =0 , 0 value means NoPorch
df['3SsnPorch'] = features['3SsnPorch']

# ScreenPorch - Screen porch area in square feet
print_stats(features, 'ScreenPorch')
# no missing values,  Mean =16.062 , Median =0 , 0 value means NoPorch 
df['ScreenPorch'] = features['ScreenPorch']

# PoolArea
print_stats(features, 'PoolArea')

# no missing values, Mean =2.251 , Median =0, 0 value means NoPool 
df['PoolArea'] = features['PoolArea']

# PoolQC
print_stats(features, 'PoolQC')

# Percentage missing = 99.66, Impute with "NoPool"
df['PoolQC'] = features['PoolQC'].fillna("NoPool")

# Fence
print_stats(features, 'Fence')
df['Fence'] = features['Fence'].fillna("NoFence")
# Percentage missing = 80.4, Impute with "NoFence" 

# MiscFeature - Miscellaneous feature not covered in other categories
print_stats(features, 'MiscFeature')

# Percentage missing = 96.4%, 2814 missing values, 6 categories,  Impute with "NoMiscFeature"
df['MiscFeature'] = features['MiscFeature'].fillna("NoMiscFeature")

# MiscVal - Value of miscellaneous feature
print_stats(features, 'MiscVal')

# no missing values, mdeian = 0, mean = 56.825
df['MiscVal'] = features['MiscVal']

# MoSold
print_stats(features, 'MoSold')

# no missing values
df['MoSold'] = features['MoSold'].astype(str)

# YrSold - Year Sold (YYYY)
print_stats(features, 'YrSold')

# No missing values
df['YrSold'] = features['YrSold'].astype(str)

# SaleType - Type of sale
print_stats(features, 'SaleType')

# 1 missing value, 10 classes, 86% values in "WD" class. Impute with that. 
df['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

# SaleCondition - Condition of sale
print_stats(features, 'SaleCondition')

# No missing values, 6 classes, 82% values in 'Normal' class
df['SaleCondition'] = features['SaleCondition']
 
#%% Export to CSV

df.to_csv("data/cleaned_df.csv", index = False, header = True)

#%%
# Also save the datatype of each variable so that 

column_dtype = {}

for column in df.columns:
    column_dtype[column] = df[column].dtype.name
    
#%% Print dictionary

for item in column_dtype:
    print(item + " : " + column_dtype[item])
    
import pickle
with open('data/column_dtype_dict.pickle', "wb") as f:
    pickle.dump(column_dtype, f, protocol = 2)
    
#%%

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
#df = pd.DataFrame()
#for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
#    df[col] = features[col].fillna('NoBSMT')

# TODO - make a dataframe with all the means and medians of all numeric fields

# =============================================================================
# # Separate categorical and numeric columns and Dummy encode
# =============================================================================

cat_vars = ['MSSubClass', 'MSZoning', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
       'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 
       'KitchenAbvGr', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 
       'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
#%%


#%%
# =============================================================================
# # Training
# =============================================================================

#%% GBM
#GBRegressor = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                           max_depth=3, max_features='sqrt',
#                                           min_samples_leaf=15, 
#                                           min_samples_split=10, 
#                                           loss='huber').fit(X_train, y_train)
#
#train_test(GBRegressor, X_train, X_test, y_train, y_test)


#%% EXPORT TO CSV

#pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('name_of_the_file.csv', index =False)



#%%


def trimm_correlated(df, threshold):
    df_corr = df.corr(method = 'pearson', min_periods = 1)
    df_not_correlated =~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype = 'bool'))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df[un_corr_idx]
    return df_out


#%%
num_df = pd.DataFrame()
    
for column in df.columns:
    if df[column].dtype != 'object':
        num_df[column] = df[column]

num_df = num_df.iloc[: , 1:]

#%%

uncorr_df = trimm_correlated(num_df, 0.5)


#%%

corr_matrix_df = df.corr()
