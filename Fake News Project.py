#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


# In[33]:


df = pd.read_csv('news.csv')


# In[34]:


df


# In[35]:


df.dtypes


# In[36]:


df.isna().sum()
# No missing values, data is cleaned


# In[37]:


#Checking for duplicates
df = df.drop_duplicates()


# In[38]:


df


# In[39]:


#Splitting dataset into training dataset and testing dataset

x_train, x_test, y_train, y_test = train_test_split(df['text'], df.label, test_size = 0.2, random_state = 0)


# In[40]:


#TfidfVectorizer used for text scenarios
#All stop words will be removed

tfidf_vect = TfidfVectorizer(stop_words = 'english')
tfidf_train = tfidf_vect.fit_transform(x_train)
tfid_test = tfidf_vect.transform(x_test)


# In[41]:


pac = PassiveAggressiveClassifier()
pac.fit(tfidf_train, y_train)

#Predict
y_pred = pac.predict(tfid_test)
score = accuracy_score(y_test , y_pred) #Get the accuracy of the classifier.
print(f'Accuracy : {round(score*100, 2)}%')


# In[ ]:





# In[42]:


#Confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#571 true negatives, 44 false positives, 40 false negatives, 612 true positives


# In[43]:


#Recall = TP / (TP + FN) 
recall = recall_score(y_test, y_pred, average='weighted')
print(recall)
print(f'Recall Score : {round(recall*100, 2)}%')


# In[44]:


#Precision = TP / (TP + FP)
precision =precision_score(y_test, y_pred, average = 'weighted')
print(precision)
print(f'Precision Score : {round(precision*100, 2)}%')


# In[45]:


#F1 score = (2 * precision * recall) / (precision + recall) - measure of model's accuracy
f1 = f1_score(y_test, y_pred, average = 'weighted')
print(f1)
print(f'F1 Score : {round(f1*100, 2)}%')


# In[ ]:





# In[ ]:





# In[ ]:




