#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler


# In[2]:


df = pd.read_csv('pokemon_data.csv')


# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.dtypes


# In[6]:


df.Legendary.unique()


# In[7]:


print(df.groupby(['Legendary']).size());


# In[8]:


df.Legendary.unique().sum


# In[9]:


df


# In[ ]:





# In[10]:


df.isna().sum()


# In[11]:


#Drop the name column
df.drop(columns=['Name', '#'], inplace = True, axis = 1)


# In[ ]:





# In[12]:


df


# In[13]:


df['Type 1'].unique()


# In[14]:


grass = df.loc[(df['Type 1']=='Grass')]
fire = df.loc[(df['Type 1']=='Fire')]
water = df.loc[(df['Type 1']=='Water')]
bug = df.loc[(df['Type 1']=='Bug')]
normal = df.loc[(df['Type 1']=='Normal')]
poison = df.loc[(df['Type 1']=='Poison')]
electric = df.loc[(df['Type 1']=='Electric')]
ground = df.loc[(df['Type 1']=='Ground')]
fairy = df.loc[(df['Type 1']=='Fairy')]
fighting = df.loc[(df['Type 1']=='Fighting')]
psychic = df.loc[(df['Type 1']=='Psychic')]
rock = df.loc[(df['Type 1']=='Rock')]
ghost =df.loc[(df['Type 1']=='Ghost')]
ice = df.loc[(df['Type 1']=='Ice')]
dragon = df.loc[(df['Type 1']=='Dragon')]
dark = df.loc[(df['Type 1']=='Dark')]
steel = df.loc[(df['Type 1']=='Steel')]
flying = df.loc[(df['Type 1']=='Flying')]


# In[ ]:





# In[15]:


print(grass.groupby(['Type 2']).size());


# In[16]:


print(fire.groupby(['Type 2']).size());


# In[17]:


print(water.groupby(['Type 2']).size());


# In[18]:


print(bug.groupby(['Type 2']).size());


# In[19]:


print(normal.groupby(['Type 2']).size());


# In[20]:


print(poison.groupby(['Type 2']).size());


# In[21]:


print(electric.groupby(['Type 2']).size());


# In[22]:


print(ground.groupby(['Type 2']).size());


# In[23]:


print(fairy.groupby(['Type 2']).size());


# In[24]:


print(fighting.groupby(['Type 2']).size());


# In[25]:


print(psychic.groupby(['Type 2']).size());


# In[26]:


print(rock.groupby(['Type 2']).size());


# In[27]:


print(ghost.groupby(['Type 2']).size());


# In[28]:


print(ice.groupby(['Type 2']).size());


# In[29]:


print(dragon.groupby(['Type 2']).size());


# In[30]:


print(dark.groupby(['Type 2']).size());


# In[31]:


print(steel.groupby(['Type 2']).size());


# In[32]:


print(flying.groupby(['Type 2']).size());


# In[33]:


df['Type 1'].unique()


# In[34]:


#If not clear, just fill it in with the type 1 value

df.loc[(df['Type 1']=='Grass') & (df['Type 2'].isna()), 'Type 2'] = grass['Type 2'].mode()[0]
df.loc[(df['Type 1']=='Fire') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Water') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Bug') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Normal') & (df['Type 2'].isna()), 'Type 2'] = normal['Type 2'].mode()[0]
df.loc[(df['Type 1']=='Poison') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Electric') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Ground') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Fairy') & (df['Type 2'].isna()), 'Type 2'] = fairy['Type 2'].mode()[0]
df.loc[(df['Type 1']=='Fighting') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Psychic') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Rock') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Ghost') & (df['Type 2'].isna()), 'Type 2'] = ghost['Type 2'].mode()[0]
df.loc[(df['Type 1']=='Ice') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Dragon') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Steel') & (df['Type 2'].isna()), 'Type 2'] = df['Type 1']
df.loc[(df['Type 1']=='Flying') & (df['Type 2'].isna()), 'Type 2'] = flying['Type 2'].mode()[0]
df['Type 2'].fillna(df['Type 1'], inplace=True)


# In[35]:


df.isna().sum()


# In[ ]:





# In[36]:


df.loc[df['Legendary'] == 'False', 'Legendary'] = 0
df.loc[df['Legendary'] == 'True', 'Legendary'] = 1
df.Legendary = df.Legendary.astype('int16')


# In[37]:


print(df.groupby(['Type 2']).size());


# In[38]:


df.dtypes


# In[39]:


c = ['Type 1', 'Type 2']
df = pd.get_dummies(df, columns = c)


# 

# In[40]:


df


# In[41]:


#Scale the values such that mean is 0 and SD is 1
c = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
ss=StandardScaler()
new_values = ss.fit(df[c].values)
new_values =ss.transform(df[c].values)
df[c] = new_values


# In[42]:


df


# In[ ]:





# In[43]:


df


# In[44]:


df = df.drop_duplicates()


# In[45]:


df


# In[46]:


#Remove the outliers for numeric columns
df = df[(np.abs(stats.zscore(df['HP'])) < 3)]
df = df[(np.abs(stats.zscore(df['Attack'])) < 3)]
df = df[(np.abs(stats.zscore(df['Defense'])) < 3)]
df = df[(np.abs(stats.zscore(df['Sp. Atk'])) < 3)]
df = df[(np.abs(stats.zscore(df['Sp. Def'])) < 3)]
df = df[(np.abs(stats.zscore(df['Speed'])) < 3)]


# In[47]:


df


# In[48]:


df


# In[49]:


df.corr(method = 'pearson')


# In[50]:


#The 5 best predictors of whether a pokemon will be considered legendary or not are Speed Attack, Speed Defense, Speed, Attack and HP respectively.


# In[51]:


df


# In[52]:


X = df.drop('Legendary',axis=1)
Y = df.Legendary
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0) 
r = RandomForestRegressor(max_depth = 20)
r.fit(train_x, train_y) 
pred_y = r.predict(test_x)


# In[53]:


pred_y


# In[54]:


pred_y = [round(x) for x in pred_y]


# In[55]:


pred_y


# In[ ]:





# In[ ]:





# In[56]:


model = XGBClassifier()
model.fit(train_x, train_y)
score = accuracy_score(test_y , pred_y)


# In[57]:


print(f'Accuracy of XGBoost Model : {round(score*100, 2)}%')


# In[58]:


#Confusion matrix
confusion_matrix = confusion_matrix(test_y, pred_y)
print(confusion_matrix)

#139 true negatives, 2 true positives, 6 false positives, 5 false negatives


# In[59]:


recall = recall_score(test_y, pred_y, average='weighted')
print(f'Recall Score : {round(recall*100, 2)}%')


# In[60]:


precision =precision_score(test_y, pred_y, average = 'weighted')
print(f'Precision Score : {round(precision*100, 2)}%')


# In[61]:


f1 = f1_score(test_y, pred_y, average = 'weighted')
print(f'F1 Score : {round(f1*100, 2)}%')


# In[62]:


lrm = r.score(test_x, test_y)
print(f'Accuracy of Logistic Regression Model : {round(lrm*100, 2)}%')


# In[63]:


dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
pred_y = dtc.predict(test_x)
dc = dtc.score(test_x, test_y)
print(f'Accuracy of Decision Tree Classifier : {round(dc*100, 2)}%')


# In[81]:


rf = RandomForestClassifier(max_depth = 100)
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)
rfs = rf.score(test_x, test_y)
print(f'Accuracy of Random Forest Classifier : {round(rfs*100, 2)}%')


# In[65]:


label = ['XGBoost', 'Random Forest', 'Decision Tree', 'Logistic Regression']
cols = ['green','blue','orange', 'red']
par = [score,rfs,dc, lrm]
plt.figure(figsize=(20,10)) 
plt.yticks(np.arange(0, 1, step=0.05))
plt.title('Accuracy of various models in predicting Legendary Pokemon', fontweight="bold", size = 30)
plt.xlabel('Model Name', fontweight="bold", size = 20)
plt.ylabel('Accuracy (between 0 and 1)', fontweight="bold", size = 20)
plt.bar(label, par, color = cols)
plt.show()


# In[66]:


#The dataset that I used contains various attributes about Pokemon such as their HP, their speed while attacking, 
#their speed while defending and their legendary status. The problem statement was what factors are the most important
#in predicting the legendary status of a particular Pokemon for all the Pokemon in the dataset.

#I first got a summary of the dataset using the head and tail features. Once I did that, 
#I checked whether there are any missing values for any of the Pokemon. I found out that 386 
#Pokemon in the dataset contained missing Type 2 values. I then researched and found out that a
#Pokemon’s missing Type 2 value can be replaced with its Type 1 value. I then checked to see 
#whether a particular Type 1 value is associated with a particular Type 2 value for every Type 1 value. 
#If I found an association, I replaced the missing Type 2 values with the associated Type 2 values 
#(based on what a particular Pokemon’s Type 1 value was). If I didn’t find an association, I replaced
#the missing Type 2 value with the Type 1 value for that particular Pokemon. I then converted the
#dataset into machine readable format by mapping false and true to 0 and 1 respectively for the 
#legendary column and using the get_dummies feature on columns that contained non-numeric values.
#I then used StandardScaler to standardize the values in such a way that the mean is 0 and the standard deviation is 1.
#I then used MinMaxScaler to ensure that all the values are between 0 and 1 and removed duplicate rows from the dataset.
#I then removed rows in which values for any particular column were more than 3 standard deviations away from the 
#mean of that particular column. I then used Pearson’s correlation method to find the most important factors in 
#predicting the legendary status of a particular Pokemon. I found out that the 5 best predictors of whether a
#pokemon will be considered legendary or not are its Speed Attack, its Speed Defense, its Speed, its Attack 
#and its HP respectively.

#I then split the Pokemon dataset into training data and testing data. I then created the models to use in order to
#predict a Pokemon’s legendary status. The models I used were decision trees, random forest, XGBoost, and logistic regression.
#I trained these models using the training dataset. After training, each model predicted whether a particular pokemon should be
#classified as legendary or not for every Pokemon in the testing dataset. Then the accuracy of each model was determined by 
#comparing the predicted legendary status for every Pokemon in the testing dataset with the actual legendary status for 
#every Pokemon in the testing dataset. The Random Forest model with a maximum depth of 100 had the highest accuracy out 
#of all the models at 96.05%, whereas the logistic regression had the lowest accuracy out of all the models at 45.22%.The
#XGBoost model had the next highest accuracy at 94.74%, followed by the decision tree classifier at 92.76%. 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




