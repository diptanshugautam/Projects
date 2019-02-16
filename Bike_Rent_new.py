# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:52:01 2019

@author: Diptanshu
"""

#!/usr/bin/env python
# coding: utf-8

#Importing required libraries
import pandas as pd
import numpy as np
import os
import statsmodels as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
import knn_impute as imp
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


#selecting working directores
os.chdir("E:\\Notes and ppts\\Edwisor\\Projects\\Project 2")



#Checking the working directory
os.getcwd()



#Importing the data
day = pd.read_csv("day.csv")
var_names = day.columns

#Excluding non-required variables
day = day.drop(['dteday','hum','windspeed','temp'], axis =1)


#Exploratory Data Analysis

#Checking the data
day.head()

#Converting required data type

day.season = day.season.astype('category')
day.yr = day.yr.astype('category')
day.mnth = day.mnth.astype('category')
day.holiday = day.holiday.astype('category')
day.workingday = day.workingday.astype('category')
day.weathersit = day.weathersit.astype('category')
day.weekday = day.weekday.astype('category')


#Selecting Numeric and Factor data
factor_data = day.select_dtypes(include = 'category')

numeric_data  = day.select_dtypes(exclude = 'category')



#Plotting Correlation Heat Map
import matplotlib.pyplot as ptt
import seaborn as sns
f, ax = ptt.subplots(figsize=(7,5))
corr = numeric_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

#Correlation Matrix
corr = numeric_data.corr()
corr.style.background_gradient()
corr

#Detect and replace with NA

#Extract quartiles
cnames = numeric_data.columns
for i in cnames:
    #calculate quartiles
    q75, q25 = np.percentile(numeric_data[i], [75 ,25])
    #Calculate Inter-quartile range
    iqr = q75 - q25
    #Calculate limits
    mini = q25 - (iqr*1.5)
    maxi = q75 + (iqr*1.5)
    #Replace with NA
    numeric_data.loc[numeric_data[i] < mini,:i] = np.nan
    numeric_data.loc[numeric_data[i] > maxi,:i] = np.nan

#imputing NAs using KNN method
for i in cnames:
    imp.knn_impute(numeric_data[i],numeric_data,5)
    
#Confirming the imputation
day.isna().sum()

#Running Anova Table
from scipy import stats
 
for i in numeric_data.columns:
    print(i)    
    F, p = stats.f_oneway(numeric_data['cnt'],numeric_data[i])
    print(p)

#Running Regression analysis
from statsmodels.formula.api  import ols

est = ols(formula = 'cnt ~ atemp+casual+registered', data = numeric_data).fit()
est.summary()

#Model Development
from sklearn import tree

#Splitting Data into Train and Test Set
train, test = train_test_split(day, test_size = 0.2)

#Decision Tree Model
regressor = DecisionTreeRegressor(random_state = 0).fit(train.iloc[:,0:11],train.iloc[:,11])

#Predicting the test set
pred = regressor.predict(test.iloc[:,0:11])


#Mean Absolute percentage error
def MAPE(actual, predicted):
    mape = np.mean((np.abs(actual - predicted)/actual))*100
    return mape

#Checking Efficiency of model
MAPE(test.iloc[:,11], pred)

#Mean Absolute Percentage Error for Decision Tree Model = 3.27%


#Random Forest Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(train.iloc[:,0:11], train.iloc[:,11])

#Predicting the test value
y_pred = regressor.predict(test.iloc[:,0:11])

#Checking Efficiency of the model
MAPE(test.iloc[:,11],y_pred)

#Mean Absolute Percentage Error for Random Forest Model = 1.63%


#Support Vector Machine Model
from sklearn.svm import SVC
regres_svm = SVC(kernel = 'linear', random_state = 0).fit(train.iloc[:,0:11], train.iloc[:,11])

#Predicting the test value
pred_svm = regres_svm.predict(test.iloc[:,0:11])

#Checking Efficiency of the model
MAPE(test.iloc[:,11],pred_svm)

#Mean Absolute Percentage Error for Support Vector Model = 1.64%







