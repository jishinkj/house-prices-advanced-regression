# House Prices: Advanced Regression Techniques

Repository for the [House Prices Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). 

We are provided with dataset that contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. We must predict the final price of each home. 

This repository contains 2 Python files - 

1. [`preprocess.py`](preprocess.py) - This Python script mainly contains the missing value imputation done on the variables. Firstly, the `SalePrice` column is dropped and the train and test dataframes are concatenated to create a dataframe with 2919 rows on which we will be doing all the preprocessing. This datafame is then exported to a CSV file which will be used in the next python script. 

2. [`main.py`](main.py) - In this Python script, we start by loading the cleaned/preprocessed dataset which is then split in the train and test sets (80:20) and then used for modeling. We have used a RandomForestRegressor. We then make predictions on the `test.csv` and export the predictions to a CSV file as per the `sample_submission.csv` provided. 

3. [`defined_functions.py`](defined_functions.py) - Contains a few custom-built functions that help to display multiple required parameters with a single function call. 
