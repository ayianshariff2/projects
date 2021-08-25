#!/usr/bin/env python
# coding: utf-8

# In[739]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier


# In[740]:


df = pd.read_csv('set2.csv')


# In[741]:


df


# In[742]:


df.head()


# In[743]:


df.dtypes


# In[744]:


#Check for null values
df.isna().sum()


# In[745]:


#Drop the unnecessary columns
df.drop(columns=['id', 'ano', 'mes', 'clues_hospital', 'fingreso', 'autoref', 'entidad', 'alc_o_municipio', 
                'fmenstrua', 'c_fecha', 'h_fingreso', 'h_fegreso', 'fecha_cierre', 'parentesco'], inplace = True, axis = 1)


# In[746]:


df.isna().sum()


# In[ ]:





# In[747]:


df.edad


# In[748]:


#Filling in the null values by column
df.isna().sum()


# In[749]:


#If there are very few null values,  replace with mode


# In[750]:


df.isna().sum()


# In[751]:


df.head()


# In[ ]:





# In[752]:


df.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[753]:


df.isna().sum()


# In[ ]:





# In[754]:



df.menarca.fillna(df.menarca.median(), inplace=True)
df.fsexual.fillna(df.fsexual.median(), inplace=True)
df.sememb.fillna(df.sememb.median(), inplace=True)
df.gesta.fillna(df.gesta.mode(), inplace=True)

df.isna().sum()


# In[755]:


#nhijos column
import seaborn as sb


# In[ ]:





# In[756]:


sb.lineplot(df['edad'], df['gesta'])


# In[ ]:





# In[757]:


df.isna().sum()


# In[ ]:





# In[758]:


df.isna().sum()


# In[759]:


print(df.groupby(['nhijos']).size())


# In[760]:


sb.lineplot(df['naborto'], df['npartos'])


# In[761]:


df.loc[(df['naborto'] == 0) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 1) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 2) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 3) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 4) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 5) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 6) & (df.npartos.isna()), 'npartos'] = 3
df.loc[(df['naborto']== 7) & (df.npartos.isna()), 'npartos'] = 2
df.loc[(df['naborto']== 8) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 9) & (df.npartos.isna()), 'npartos'] = 0
df.loc[(df['naborto']== 10) & (df.npartos.isna()), 'npartos'] = 3


# In[ ]:





# In[762]:


df.isna().sum()


# In[763]:


df['c_num'].fillna(df.c_num.median(),inplace=True)


# In[764]:


df.isna().sum()


# In[765]:


df.drop(columns=['p_consent', 's_complica', 'c_dolor', 'tanalgesico', 'cconsejo', 'edocivil_descripcion', 'edad', 'procile'], inplace = True, axis = 1)


# In[766]:


df.isna().sum()


# In[767]:


df.isna().sum()


# In[768]:


#Normalization - removing redundancies, anomalies, inconsistencies
df


# In[769]:


df = df.drop_duplicates()


# In[770]:


df


# In[ ]:





# In[771]:


df.head(10)


# In[772]:


df


# In[ ]:





# In[773]:


df.nhijos.fillna(df.nhijos.mode(), inplace=True)


# In[ ]:





# In[774]:


df.naborto.fillna(df.naborto.mode()[0], inplace=True)


# In[775]:


df.loc[(df['naborto'] == 0) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 1) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 2) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 3) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 4) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 5) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 6) & (df.npartos.isna()), 'npartos'] = 3
df.loc[(df['naborto']== 7) & (df.npartos.isna()), 'npartos'] = 2
df.loc[(df['naborto']== 8) & (df.npartos.isna()), 'npartos'] = 1
df.loc[(df['naborto']== 9) & (df.npartos.isna()), 'npartos'] = 0
df.loc[(df['naborto']== 10) & (df.npartos.isna()), 'npartos'] = 3


# In[776]:


df.ncesarea.fillna(df.ncesarea.mode()[0], inplace=True)


# In[777]:


df.nile.fillna(df.nile.mode()[0], inplace=True)


# In[778]:


df.p_semgest.fillna(df.p_semgest.mode()[0],inplace=True)


# In[779]:


df['p_diasgesta'].fillna(df.p_diasgesta.median(),inplace=True)


# In[780]:


df.gesta.fillna(df.gesta.mode()[0], inplace=True)
df.nhijos.fillna(df.nhijos.mode()[0], inplace=True)
df.ocupacion.fillna('NO ESPECIFICADO',inplace=True)
df.religion.fillna(df.religion.mode()[0],inplace=True)
df.anticonceptivo.fillna(df.anticonceptivo.mode()[0], inplace=True)
df.menarca.fillna(df.menarca.median(), inplace=True)
df.fsexual.fillna(df.fsexual.median(), inplace=True)
df.sememb.fillna(df.sememb.median(), inplace=True)
df.desc_derechohab.fillna(df.desc_derechohab.mode()[0],inplace=True)
df.nivel_edu.fillna('PREPARATORIA', inplace = True)
df['motiles'].fillna('OTRA',inplace=True)
df['c_num'].fillna(df.c_num.median(),inplace=True)
df['desc_servicio'].fillna(df.desc_servicio.mode()[0],inplace=True)
df['p_diasgesta'].fillna(df.p_diasgesta.median(),inplace=True)
df['panticoncep'].fillna(df.panticoncep.mode()[0],inplace=True)
df.resultado_ile.fillna(df.resultado_ile.mode()[0], inplace=True)
df.procile_simplificada.fillna(df.procile_simplificada.mode()[0],inplace=True)
df.consejeria.fillna(df.consejeria.mode()[0], inplace=True)
df.isna().sum()


# In[781]:


#Remove outliers for numeric values
df = df[(np.abs(stats.zscore(df['menarca'])) < 3)]
df = df[(np.abs(stats.zscore(df['fsexual'])) < 3)]
df = df[(np.abs(stats.zscore(df['sememb'])) < 3)]
df = df[(np.abs(stats.zscore(df['nhijos'])) < 3)]
df = df[(np.abs(stats.zscore(df['gesta'])) < 3)]
df = df[(np.abs(stats.zscore(df['naborto'])) < 3)]
df = df[(np.abs(stats.zscore(df['npartos'])) < 3)]
df = df[(np.abs(stats.zscore(df['ncesarea'])) < 3)]
df = df[(np.abs(stats.zscore(df['nile'])) < 3)]
df = df[(np.abs(stats.zscore(df['c_num'])) < 3)]
df = df[(np.abs(stats.zscore(df['p_semgest'])) < 3)]
df = df[(np.abs(stats.zscore(df['p_diasgesta'])) < 3)]
df.head(10)


# In[782]:


df.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[783]:


df


# In[784]:



df.loc[df['c_num'] < 2, 'c_num'] = 0
df.loc[df['c_num'] >= 2, 'c_num'] = 1


# In[785]:


print(df.groupby(['c_num']).size())


# In[786]:


df.nivel_edu.unique()


# In[787]:


df.dtypes


# In[788]:


#Consider everything, type of data. According to type, do standardization
from sklearn.preprocessing import StandardScaler

#Classify data according to the type. Do standardization according to the dataset
#Change the values in the column, mapping. Use mapping for large number of unique values, or binary values.

nivel_edu = {'SIN ACCESO A LA EDUCACION FORMAL' : 0, 'PREPARATORIA' : 1, 'PRIMARIA' : 2, 'SECUNDARIA' : 3, 'LICENCIATURA' : 4,
             'MAESTRIA' : 5, 'DOCTORADO' : 6}
df['nivel_edu'] = df['nivel_edu'].map(nivel_edu)

resultado_ile = {'COMPLETA' : 0, 'OTRO' : 1}
df['resultado_ile'] = df['resultado_ile'].map(resultado_ile)


consejeria = {'NO' : 0, 'SI' : 1}
df['consejeria'] = df['consejeria'].map(consejeria)
#Yes or no, you can use mapping



# In[789]:


c = ['religion', 'ocupacion', 'desc_derechohab', 'anticonceptivo', 'motiles', 'desc_servicio', 'panticoncep', 'procile_simplificada']
df = pd.get_dummies(df, columns = c)
df


# In[ ]:





# In[790]:


#Scale only the continious variables
c = ['menarca', 'fsexual', 'sememb', 'nhijos', 'gesta', 'naborto', 'npartos', 'ncesarea', 'nile', 'c_num', 
        'p_semgest', 'p_diasgesta']
ss=StandardScaler()
ss = ss.fit(df[c].values)
new_values=ss.transform(df[c].values)
new_values
df[c] = new_values


# In[791]:


df


# In[ ]:





# In[792]:


df.head(10)


# In[793]:


df


# In[794]:


df.isna().sum()


# In[ ]:





# In[795]:


df = df.astype(int)


# In[796]:


df.dtypes


# In[797]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


X = df.drop('naborto',axis=1)
Y = df.naborto
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0) #random state = 0, jumbles data
r = RandomForestClassifier(max_depth = 15) #creating object - max depth, height of tree
r.fit(train_x, train_y) 
fimpvalues = r.feature_importances_ #getting an array, got helper to get features
type(r.feature_importances_)


# In[798]:



bars = 10 #prints 10 most important features, which we can pick
indices = np.argsort(fimpvalues)[::-1][:bars]
yindex = np.arange(bars)
plt.yticks(yindex, X.columns.values[indices])
plt.barh(yindex,fimpvalues[indices])


# In[799]:



bars = 15 #prints 15 most important features, which we can pick
indices = np.argsort(fimpvalues)[::-1][:bars]
yindex = np.arange(bars)
plt.yticks(yindex, X.columns.values[indices])
plt.barh(yindex,fimpvalues[indices])


# In[800]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(df),
            columns=df.columns, index=df.index)

print(normalized.head(10))
print(normalized.tail(10))


# In[801]:


dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
pred_y = dtc.predict(test_x) 


# In[ ]:





# In[802]:


pred_y


# In[803]:


#Decision tree score
a = dtc.score(test_x, test_y)


# In[804]:


#Random forest score ,  depth of 100

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 100)
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)
b = rf.score(test_x, test_y)


# In[805]:


#Logistic regression score
c = r.score(test_x, test_y)


# In[806]:


#Random forest score ,  depth of 10

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 10)
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)
d = rf.score(test_x, test_y)


# In[807]:


#Random forest score ,  depth of 50

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 50)
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)
e = rf.score(test_x, test_y)


# In[808]:


#Random forest score ,  depth of 75

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 75)
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)
f = rf.score(test_x, test_y)


# In[809]:


#Visually plotting the accuracy of all the algorithms

import matplotlib.pyplot as plt

label = ['Decision Tree', 'Random Forest maximum depth of 100', 'Logistic Regression', 'Random Forest maximum depth of 10', 'Random Forest depth of 50', 'Random Forest maximum depth of 75' ]
ya = [a, b, c, d, e, f]
nc = ['green','blue','gray','yellow','red', 'purple']
plt.figure(figsize=(20,10)) 
plt.yticks(np.arange(0, 1, step=0.05))
plt.title('Accuracy of various algorithms', fontweight="bold", size = 30)
plt.xlabel('Algorithm Name', fontweight="bold", size = 20)
plt.ylabel('Accuracy (between 0 and 1)', fontweight="bold", size = 20)
plt.bar(label, ya, color = nc)
plt.show()


# In[ ]:





# In[819]:


#Line plot of the accuracy of all the algorithms

import matplotlib.pyplot as plt

label = ['Decision Tree', 'Random Forest maximum depth of 100', 'Logistic Regression', 'Random Forest maximum depth of 10', 'Random Forest depth of 50', 'Random Forest maximum depth of 75' ]
ya = [a, b, c, d, e, f]
nc = ['green','blue','gray','yellow','red', 'purple']
plt.figure(figsize=(20,10)) 
plt.yticks(np.arange(0, 1, step=0.05))
plt.title('Accuracy of various algorithms', fontweight="bold", size = 30)
plt.xlabel('Algorithm Name', fontweight="bold", size = 20)
plt.ylabel('Relative Accuracy', fontweight="bold", size = 20)
plt.plot(label, ya)
plt.show()


# In[811]:


print(a)


# In[812]:


print(b)


# In[813]:


print(c)


# In[814]:


print(d)


# In[815]:


print(e)


# In[816]:


print(f)


# In[817]:


#I first split the ILE dataset into training data and testing data. I then created the models to 
#use in order to predict the number of abortions - a decision tree, random forests, and a logistic regression model -
#and trained them using the training dataset. After training, each model predicted 
#the number of abortions on the testing dataset. Then the accuracy of each model was determined by comparing the 
#predicted number of abortions for each row in the testing dataset to the actual number of abortions for each row 
#in the testing dataset. The random forest with a maximum depth 
#(maximum distance between the root node and a leaf node) of 100 and 50 had the highest accuracy out of all the models
#at 89.3%, whereas the decision tree had the lowest accuracy out of all the models at 85.99%.
#The random forest with a maximum depth of 75 had the next highest accuracy (at 89.24%), followed by the logistic regression 
#model (at 88.76%), followed by the random forest with a maximum depth of 10 (at 88%). 
#The five most beneficial factors in predicting the number of abortions (in order) are gesta, the number 
#of children they have, the number of illegal abortions, npartos and the number of c-sections. 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




