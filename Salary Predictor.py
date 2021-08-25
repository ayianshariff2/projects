#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[123]:


df = pd.read_csv('adult_train.csv')


# In[124]:


df


# In[125]:


df.isna().sum()


# In[126]:


df.drop(columns=['fnlwgt'], inplace = True, axis = 1)
c = ['workclass', 'occupation', 'relationship', 'race', 'native-country', 'marital-status', 'sex']
df = pd.get_dummies(df, columns = c)


# In[127]:


df.education.unique()


# In[128]:


df.salary.unique()


# In[ ]:





# In[129]:


#Mapping counts under standardization
education = {' Preschool' : 0, ' 1st-4th' : 1, ' 5th-6th' : 2, ' 7th-8th' : 3, ' 9th' : 4, ' 10th' : 5, ' 11th' : 6, ' 12th' : 7,
            ' HS-grad' : 8, ' Some-college' : 9, ' Assoc-voc' : 10, ' Assoc-acdm' : 10, ' Bachelors' : 11, ' Masters' : 12, 
             ' Doctorate' : 13,  ' Prof-school' : 14}
df['education'] = df['education'].map(education)
salary = { ' <=50K' : 0, ' >50K' : 1}
df['salary'] = df['salary'].map(salary)


# In[130]:


df


# In[131]:


scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df),
            columns=df.columns, index=df.index)


# In[ ]:





# In[ ]:





# In[ ]:





# In[132]:


#Drop duplicate columns
df = df.drop_duplicates()


# In[133]:


#Remove the outliers
df = df[(np.abs(stats.zscore(df['age'])) < 3)]
df = df[(np.abs(stats.zscore(df['education'])) < 3)]
df = df[(np.abs(stats.zscore(df['education-num'])) < 3)]
df = df[(np.abs(stats.zscore(df['capital-gain'])) < 3)]
df = df[(np.abs(stats.zscore(df['capital-loss'])) < 3)]
df = df[(np.abs(stats.zscore(df['hours-per-week'])) < 3)]


# In[134]:


df


# In[135]:


#Feature importances
X = df.drop('salary',axis=1)
Y = df.salary
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0) #random state = 0, jumbles data
r = RandomForestClassifier(max_depth = 20) 
r.fit(train_x, train_y) 
fimpvalues = r.feature_importances_
type(r.feature_importances_)


# In[136]:


bars = 10 #prints 10 most important features, which we can pick
indices = np.argsort(fimpvalues)[::-1][:bars]
yindex = np.arange(bars)
plt.yticks(yindex, X.columns.values[indices])
plt.barh(yindex,fimpvalues[indices])


# In[137]:


#5 most important predictors of salary are capital-gain, age, marital-status, education and hrs per week


# In[138]:


r = RandomForestRegressor(max_depth = 20)
r.fit(train_x, train_y) 
pred_y = r.predict(test_x)
pred_y = [round(x) for x in pred_y]
model = XGBClassifier(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.15,
                max_depth = 5, alpha = 10, n_estimators = 50)
model.fit(train_x, train_y)
score = accuracy_score(test_y , pred_y)


# In[139]:


print(f'Accuracy of XGBoost Model : {round(score*100, 2)}%')


# In[140]:


confusion_matrix = confusion_matrix(test_y, pred_y)
print(confusion_matrix)
#4231 true negatives, 788 true positives, 284 false positives, 597 false negatives


# In[141]:


recall = recall_score(test_y, pred_y, average='weighted')
print(f'Recall Score : {round(recall*100, 2)}%')


# In[142]:


precision =precision_score(test_y, pred_y, average = 'weighted')
print(f'Precision Score : {round(precision*100, 2)}%')


# In[143]:


f1 = f1_score(test_y, pred_y, average = 'weighted')
print(f'F1 Score : {round(f1*100, 2)}%')


# In[144]:


lrm = r.score(test_x, test_y)
print(f'Accuracy of Logistic Regression Model : {round(lrm*100, 2)}%')


# In[196]:


dtc = DecisionTreeClassifier(criterion="gini", splitter = "best", max_depth=10)
dtc.fit(train_x, train_y)
pred_y = dtc.predict(test_x)
dc = dtc.score(test_x, test_y)
print(f'Accuracy of Decision Tree Classifier : {round(dc*100, 2)}%')


# In[193]:


rf = RandomForestClassifier(bootstrap=True, criterion='entropy',
            min_impurity_decrease=0.0,
            min_samples_leaf=10, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,verbose=0)
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)
rfs = rf.score(test_x, test_y)
print(f'Accuracy of Random Forest Classifier : {round(rfs*100, 2)}%')


# In[197]:


label = ['XGBoost', 'Random Forest', 'Decision Tree', 'Logistic Regression']
cols = ['teal','orange','blue', 'green']
par = [score,rfs,dc, lrm]
plt.figure(figsize=(20,10)) 
plt.yticks(np.arange(0, 1, step=0.05))
plt.title('Accuracy of various models in predicting income level', fontweight="bold", size = 30)
plt.xlabel('Model Name', fontweight="bold", size = 20)
plt.ylabel('Accuracy (between 0 and 1)', fontweight="bold", size = 20)
plt.bar(label, par, color = cols)
plt.show()


# In[ ]:




