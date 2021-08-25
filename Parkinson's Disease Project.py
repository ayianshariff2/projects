#!/usr/bin/env python
# coding: utf-8

# In[414]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[415]:


df = pd.read_csv('parkinsons.data')


# In[416]:


df


# In[417]:


df.isna().sum()


# In[418]:


df = df.drop_duplicates()


# In[ ]:





# In[419]:


df


# In[420]:


df.dtypes


# In[421]:


#Get features and labels. Status column is the target label
labels=df.loc[:,'status'].values
features=df.loc[:,df.columns!='status'].values[:,1:]

print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# In[422]:


df.status


# In[423]:


#Standard Scaler
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features) #Transforms data values so that every data value is between -1 and 1 without loss of ordering
y = labels


# In[424]:


x


# In[425]:


y


# In[426]:


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[ ]:





# In[427]:


#Train the model
model = XGBClassifier()
model.fit(train_x, train_y)


# In[428]:


#Test the model on the test data
pred_y = model.predict(test_x)


# In[429]:


#Evaluate the accuracy of the XGBoost model
score = accuracy_score(test_y , pred_y)
print(f'Accuracy of XGBoost Model : {round(score*100, 2)}%')


# In[430]:


#Confusion matrix
confusion_matrix = confusion_matrix(test_y, pred_y)
print(confusion_matrix)

#10 true negatives, 0 false positives, 2 false negatives, 27 true positives


# In[431]:


#Recall = TP / (TP + FN) 
recall = recall_score(test_y, pred_y, average='weighted')
print(recall)
print(f'Recall Score : {round(recall*100, 2)}%')


# In[432]:


#Precision = TP / (TP + FP)
precision =precision_score(test_y, pred_y, average = 'weighted')
print(precision)
print(f'Precision Score : {round(precision*100, 2)}%')


# In[433]:


#F1 score = (2 * precision * recall) / (precision + recall) - measure of model's accuracy
f1 = f1_score(test_y, pred_y, average = 'weighted')
print(f1)
print(f'F1 Score : {round(f1*100, 2)}%')


# In[434]:


#Implement the random forest model and evaluate it's accuracy
rf = RandomForestClassifier(max_depth = 50)
rf.fit(train_x, train_y)
pred = rf.predict(test_x)
rfscore = rf.score(test_x, test_y)
print(f'Accuracy of Random Forest Model : {round(rfscore*100, 2)}%')


# In[435]:


#Implement the decision tree model and evaluate it's accuracy
dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
pred = dtc.predict(test_x)
dtcs= dtc.score(test_x, test_y)
print(f'Accuracy of Decision Tree Model : {round(dtcs*100, 2)}%')


# In[436]:


#Evaluating the accuracy of the logistic regression model
lm = r.score(test_x, test_y)
print(f'Accuracy of Logistic Regression Model : {round(lm*100, 2)}%')


# In[437]:


label = ['XGBoost', 'Random Forest', 'Decision Tree', 'Logistic Regression']
alg = [score,rfscore,dtcs, lm]
cols = ['blue','yellow','red', 'green']
plt.figure(figsize=(20,10)) 
plt.yticks(np.arange(0, 1, step=0.05))
plt.title('Accuracy of various models in predicting Parkinson disease', fontweight="bold", size = 30)
plt.xlabel('Model Name', fontweight="bold", size = 20)
plt.ylabel('Accuracy (between 0 and 1)', fontweight="bold", size = 20)
plt.bar(label, alg, color = cols)
plt.show()


# In[ ]:




