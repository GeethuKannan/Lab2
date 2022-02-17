#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cancer_df = pd.read_csv('C:/Users/user/Downloads/TASK/dataset.csv')


# In[3]:


cancer_df.shape


# In[4]:


display(cancer_df)


# In[5]:


cancer_df=cancer_df.drop(columns=['id'])


# ## EDA : Exploratory Data Analysis

# In[6]:


#Check for missing values in the dataset
cancer_df.isnull().sum()


# As we can see from the above output there are no missing values in the dataset

# In[7]:


#Class Balance
print('Class Split')
print(cancer_df['diagnosis'].value_counts())
cancer_df['diagnosis'].value_counts().plot.bar(figsize=(10,4),title='Classes Split for Dataset')
plt.xlabel('Classes')
plt.ylabel('Count')


# There is an imbalance in the data under the two classes B and M. There are 350 records under class B and only half of them under class M.

# In[8]:


#Duplicate values check
Dup_df = cancer_df[cancer_df.duplicated()]

print("Evidence of {} duplicate rows in the dataset".format(Dup_df.shape[0]))
Dup_df


# ## Outlier Detection

# In[9]:


#Boxplot Visualization
plt.figure(figsize=(15,10))
sns.boxplot(data=cancer_df)


# there are many ouliers in the dataset as we can see from the box plot above

# In[10]:


#Tukey Method

# Import required libraries
from collections import Counter

# Outlier detection 
def detect_outliers(df,n,features):
    
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# List of Outliers
Outliers_to_drop = detect_outliers(cancer_df.drop('diagnosis', axis=1),0,list(cancer_df.drop('diagnosis', axis=1)))
cancer_df.drop('diagnosis', axis=1).loc[Outliers_to_drop]


# Above are the list of outliers in the dataset.

# In[11]:


event={'B':0,'M':1}


# In[12]:


cancer_df.diagnosis=[event[item] for item in cancer_df.diagnosis]
print(cancer_df)


# In[13]:


X = cancer_df.drop(columns=['diagnosis']).values
y = cancer_df['diagnosis'].values


# ## Test and Train data split

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[15]:


print('Shape of X_train=>',X_train.shape)
print('Shape of X_test=>',X_test.shape)
print('Shape of Y_train=>',y_train.shape)
print('Shape of Y_test=>',y_test.shape)


# ## Random Forest Classification

# In[17]:


# Building  Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
rfc = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, random_state = 20)
rfc.fit(X_train, y_train)

# Evaluating on Training set
rfc_pred_train = rfc.predict(X_train)
print('Training Set Evaluation F1-Score=>',f1_score(y_train,rfc_pred_train))


# In[18]:


# Evaluating on Test set
rfc_pred_test = rfc.predict(X_test)
print('Testing Set Evaluation F1-Score=>',f1_score(y_test,rfc_pred_test))


# In[ ]:





# In[ ]:





# In[ ]:




