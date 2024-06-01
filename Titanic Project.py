#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

df = pd.read_csv('titanic3.csv')
df.head(10)


# In[3]:


df.isna().sum()


# In[4]:


df = df.dropna(thresh=11)


# In[5]:


df.drop(columns=['cabin','body','name','boat','home.dest','ticket'],inplace=True,axis=1)


# In[6]:


df.loc[df['age'].isnull(),'age']=int(df.age.mean())


# In[7]:


df.embarked.fillna('S',inplace=True)


# In[ ]:





# In[8]:


x = df.copy()


# In[9]:


sex = {'male':1,'female':0}
x['sex'] = x['sex'].map(sex)


# In[10]:


x


# In[11]:


x.age = x.age.astype('int8')


# In[12]:


min_age = 2
x.drop(x[x['age']<min_age].index,inplace=True)


# In[13]:


columns=['pclass','sibsp','embarked','parch']
x=pd.get_dummies(x,columns=columns)
x.head(10)


# In[14]:


x.dtypes


# In[15]:


from sklearn.preprocessing import StandardScaler


cols = ['age','fare']
ss=StandardScaler()
ss = ss.fit(x[cols].values)
new_values=ss.transform(x[cols].values)
new_values


# In[16]:


print(x[cols].values)


# In[17]:


len(new_values[:,1])


# In[18]:


import matplotlib.pyplot as plt
plt.plot(new_values[:,0],  color = 'blue')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare paid')
plt.plot(new_values[:,1], color = 'red')

#


# In[19]:


#Draw the histogram for the columns after applying standardization, descriptive analysis, why are we using standard scale.

plt.hist(new_values[:,0], color = 'blue')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare paid')
plt.hist(new_values[:,1],  color = 'red')
plt.legend()


# In[20]:


import statistics
z = statistics.mean(new_values[:,0])
l = statistics.mean(new_values[:,1])
print(z)
print(l)

#Distributed in such a way that mean is 0 and variance is 1


# In[21]:


#Take the new standardized values as part of the data frame
x[cols] = new_values


# In[22]:


print(new_values)


# In[23]:


print(x)


# In[24]:


import seaborn as sns
#Star mark means still running


# In[25]:


#sns.pairplot(x)

X = x.drop('survived',axis=1)
Y = x.survived


# In[26]:


#Can't see line plot or box plot or histogram with sns pairplot. But if necessary can use seaborn


# In[27]:





# In[28]:


#Feature selection
from sklearn.model_selection import train_test_split




# In[29]:


Y


# In[30]:


X


# In[31]:


#We are now supposed to split the data into training and testing
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0) #random state = 0, jumbles data
from sklearn.ensemble import RandomForestRegressor
r = RandomForestRegressor(max_depth = 15) #creating object - max depth, height of tree
r.fit(train_x, train_y) 
fimpvalues = r.feature_importances_ #getting an array, got helper to get features
type(r.feature_importances_)


# In[32]:


fimpvalues #23 values - 1 value for each column - calculating importance

#which column most related to the feature


# In[33]:


import numpy as np
bars = 10 #prints 10 most important features, which we can pick
indices = np.argsort(fimpvalues)[::-1][:bars]
yindex = np.arange(bars)
plt.yticks(yindex, X.columns.values[indices])
plt.barh(yindex,fimpvalues[indices])


# In[34]:


df.dtypes


# In[35]:


df.dtypes
#Replace female with 0, replace male with 1. Drop the embarked column
df.drop(columns=['embarked'],inplace=True,axis=1)


# In[36]:


df['sex'] = df['sex'].replace('female', '0')
df['sex'] = df['sex'].replace('male', '1')
df.sex = df.sex.astype('int64')


# In[37]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(df),
            columns=df.columns, index=df.index)


# In[38]:


print(normalized.head(10))


# In[39]:


print(normalized.tail(10))


# In[40]:


from sklearn.tree import DecisionTreeClassifier


# In[41]:


dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
pred_y = dtc.predict(test_x) #Dtc - new decision tree model - use that rather


# In[42]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_y, pred_y)
print(confusion_matrix)


# In[43]:


pred_y


# In[44]:


#Implement a random forest
from sklearn.ensemble import RandomForestClassifier


# In[45]:


rf = RandomForestClassifier(max_depth = 100)
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)


# In[ ]:





# In[46]:


pred_y


# In[47]:


#Random forest score
rf.score(test_x, test_y)


# In[48]:


#Decision tree score
dtc.score(test_x, test_y)


# In[49]:


#Logistic regression score
r.score(test_x, test_y)


# In[50]:


# The less the depth, the higher scoreIN THIS CASE. Which max depth you can use. play with those parameters, trial and error

#Random forest model is better for the problem statement

