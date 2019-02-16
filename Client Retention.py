#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Load Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
import knn_impute as ki


# In[3]:


#Set working directory
os.chdir("E:\\Notes and ppts\\Edwisor\\Projects\\Project 3")


# In[167]:


#Loading Data
train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")


# In[168]:


#Appending train and test data
data = train.append(test)


# In[169]:


data.head()


# In[170]:


#Data Preprocessing
# Encoding the Independent categorical Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
data.iloc[:, 0] = labelencoder_X.fit_transform(data.iloc[:, 0])
data.iloc[:, 2] = labelencoder_X.fit_transform(data.iloc[:, 2])
data.iloc[:, 3] = labelencoder_X.fit_transform(data.iloc[:, 3])
data.iloc[:, 4] = labelencoder_X.fit_transform(data.iloc[:, 4])
data.iloc[:, 5] = labelencoder_X.fit_transform(data.iloc[:, 5])
data.iloc[:, 20] = labelencoder_X.fit_transform(data.iloc[:, 20])

#Converting the respective data type to categorical
data.state = data.state.astype('category')
data['international plan'] = data['international plan'].astype('category')
data['voice mail plan'] = data['voice mail plan'].astype('category')
data.Churn = data.Churn.astype('category')
data['area code'] = data['area code'].astype('category')
data['phone number'] = data['phone number'].astype('category')


# In[171]:


data.dtypes


# In[301]:


#Checking for missing value
data.isnull().sum()


# In[145]:


#Visualizing box plot
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(data['total day charge'])


# In[172]:


#Extracting Numeric and Categorical data
numeric_data = data.select_dtypes(exclude = ['category'])
factor_data = data.select_dtypes(include=['category'])


# In[173]:


cnames = numeric_data.columns


# In[174]:


#Checking Outliers and replacing them with NAs
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


# In[176]:


#imputing NAs using KNN method
for i in cnames:
    ki.knn_impute(numeric_data[i],numeric_data,7)


# In[177]:


#Confirming the imputation
data.isna().sum()


# In[15]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corel = numeric_data.corr()

#Ploting the correlation analysis
sns.heatmap(corel, mask=np.zeros_like(corel, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[196]:


cate_var = data.columns


# In[197]:


#Check Chi Square Analysis
for i in cate_var:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data['Churn'], data[i]))
    print(p, chi2)


# In[58]:


#combining numeric and factor data
data = pd.concat([numeric_data, factor_data],axis = 1)


# In[180]:


#Dimensionality reduction
data = data.drop(['area code','phone number','total intl minutes','total day minutes','total eve minutes','total night minutes'],axis = 1)


# In[181]:


#Splitting the train and test data in original ratio
train = data.iloc[0:3333,:]
test = data.iloc[3333:5000,:]


# In[182]:


#Model Development
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[183]:


data.shape


# In[184]:


#Splitting the dependent and independent from test and train set
x_train = train.values[:,0:14]
y_train = train.values[:,14]
x_test = test.values[:,0:14]
y_test = test.values[:,14]


# In[185]:


#converting the data type of the dependent variable train and test data
y_train = pd.Series(y_train).astype('category').values
y_test = pd.Series(y_test).astype('category').values


# In[186]:


#Decision Tree Model
DT_model = tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)

#Predicting the test dataset
DT_Predictions = DT_model.predict(x_test)


# In[271]:


#Confusion matrix
from sklearn.metrics import confusion_matrix 
CM1 = confusion_matrix(y_test, DT_Predictions)
CM = pd.crosstab(y_test, DT_Predictions)

#Saving True-Positive, True-Negative, False-Positive, False-Negative
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

#Results
#Accuracy: 92.14
#FNR: 32.58


# In[272]:


CM


# In[ ]:


#Building Random Forest Model
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 250).fit(x_train, y_train)


# In[189]:


#Predicting the dependent variable
RF_Predictions = RF_model.predict(x_test)


# In[269]:


#Confusion Matrix
# from sklearn.metrics import confusion_matrix 
CM1 = confusion_matrix(y_test, RF_Predictions)
CM = pd.crosstab(y_test, RF_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

#Accuracy: 95.68
#FNR: 29.46


# In[191]:


#Building Naive Bayes Model
from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(x_train, y_train)


# In[192]:


#predict test cases
NB_Predictions = NB_model.predict(x_test)


# In[273]:


#Confusion matrix
CM = pd.crosstab(y_test, NB_Predictions)
CM1 = confusion_matrix(y_test, NB_Predictions)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

#Accuracy: 85.84
#FNR: 60.26


# In[ ]:




