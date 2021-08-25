#!/usr/bin/env python
# coding: utf-8

# In[572]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# In[573]:


df = pd.read_csv('set2.csv')


# In[574]:


df


# In[575]:


df.head()


# In[576]:


len(df)


# In[577]:


#Check for null values
df.isna().sum()


# In[578]:


#Drop the unnecessary columns
df.drop(columns=['id', 'ano', 'mes', 'clues_hospital', 'fingreso', 'autoref', 'entidad', 'alc_o_municipio', 
                'fmenstrua', 'c_fecha', 'h_fingreso', 'h_fegreso', 'fecha_cierre'], inplace = True, axis = 1)


# In[579]:


df.isna().sum()


# In[580]:


#Group age into bins. check distribution first. most people seem to be in their 20's and 30's
bins = [0, 18, 22, 26, 30, 34, 40, 50, 60]
labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8']
df['edad'] = pd.cut(df['edad'], bins=bins, labels=labels, right=False)


# In[581]:


df.edad


# In[582]:


#Filling in the null values by column
df.isna().sum()


# In[583]:


#If there are very few null values,  replace with mode
df.edad.fillna(df.edad.mode()[0],inplace=True)
df.procile.fillna(df.procile.mode()[0],inplace=True)
df.procile_simplificada.fillna(df.procile_simplificada.mode()[0],inplace=True)
df.p_semgest.fillna(df.p_semgest.mode()[0],inplace=True)


# In[584]:


df.isna().sum()


# In[585]:


df.head()


# In[ ]:





# In[586]:


df.isna().sum()


# In[587]:


df.desc_derechohab.fillna(df.desc_derechohab.mode()[0],inplace=True)


# In[ ]:





# In[588]:


#Cleaning education column
#Different approach

df.loc[(df['ocupacion']=='TRABAJADORA DEL HOGAR NO REMUNERADA') & (df.nivel_edu.isna()), 'nivel_edu'] = 'SECUNDARIA'
df.loc[(df['ocupacion']=='ESTUDIANTE') & (df.nivel_edu.isna()), 'nivel_edu'] = 'SECUNDERIA'
df.loc[(df['ocupacion']=='EMPLEADA') & (df.nivel_edu.isna()), 'nivel_edu'] = 'PREPARATORIA'
df.loc[(df['ocupacion']=='DESEMPLEADA') & (df.nivel_edu.isna()), 'nivel_edu'] = 'PREPARATORIA'
df.loc[(df['ocupacion']=='NO ESPECIFICADO') & (df.nivel_edu.isna()), 'nivel_edu'] = 'PREPARATORIA'
df.nivel_edu.fillna('PREPARATORIA', inplace = True)


# In[589]:


df.ocupacion.unique()


# In[590]:


df.isna().sum()


# In[591]:


df.ocupacion.fillna('NO ESPECIFICADO',inplace=True)


# In[592]:


df.religion.fillna(df.religion.mode()[0],inplace=True)
df.parentesco.fillna(df.parentesco.mode()[0], inplace=True)
df.anticonceptivo.fillna(df.anticonceptivo.mode()[0], inplace=True)
df.menarca.fillna(df.menarca.median(), inplace=True)
df.fsexual.fillna(df.fsexual.median(), inplace=True)
df.sememb.fillna(df.sememb.median(), inplace=True)

df.isna().sum()


# In[593]:


#nhijos column
import seaborn as sb
df.loc[(df['edad']== 'G1') & (df.nhijos.isna()), 'nhijos'] = 0
df.loc[(df['edad']== 'G2') & (df.nhijos.isna()), 'nhijos'] = 0
df.loc[(df['edad']== 'G3') & (df.nhijos.isna()), 'nhijos'] = 0
df.loc[(df['edad']== 'G4') & (df.nhijos.isna()), 'nhijos'] = 1
df.loc[(df['edad']== 'G5') & (df.nhijos.isna()), 'nhijos'] = 3
df.loc[(df['edad']== 'G6') & (df.nhijos.isna()), 'nhijos'] = 9
df.loc[(df['edad']== 'G7') & (df.nhijos.isna()), 'nhijos'] = 10
df.loc[(df['edad']== 'G8') & (df.nhijos.isna()), 'nhijos'] = 0


# In[ ]:





# In[594]:


sb.lineplot(df['edad'], df['gesta'])


# In[595]:



df.loc[(df['edad']== 'G1') & (df.gesta.isna()), 'gesta'] = 1
df.loc[(df['edad']== 'G2') & (df.gesta.isna()), 'gesta'] = 1
df.loc[(df['edad']== 'G3') & (df.gesta.isna()), 'gesta'] = 2
df.loc[(df['edad']== 'G4') & (df.gesta.isna()), 'gesta'] = 2
df.loc[(df['edad']== 'G5') & (df.gesta.isna()), 'gesta'] = 3
df.loc[(df['edad']== 'G6') & (df.gesta.isna()), 'gesta'] = 3
df.loc[(df['edad']== 'G7') & (df.gesta.isna()), 'gesta'] = 4
df.loc[(df['edad']== 'G8') & (df.gesta.isna()), 'gesta'] = 3


# In[596]:


df.isna().sum()


# In[597]:


df.naborto.fillna(df.naborto.mode()[0], inplace=True)
df.ncesarea.fillna(df.ncesarea.mode()[0], inplace=True)
df.nile.fillna(df.nile.mode()[0], inplace=True)
df.consejeria.fillna(df.consejeria.mode()[0], inplace=True)
df.resultado_ile.fillna(df.resultado_ile.mode()[0], inplace=True)
df.edocivil_descripcion.fillna(df.edocivil_descripcion.mode()[0],inplace=True)


# In[598]:


df.isna().sum()


# In[599]:


df


# In[600]:


sb.lineplot(df['naborto'], df['npartos'])


# In[601]:


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





# In[602]:


df.isna().sum()


# In[603]:


df.motiles.unique()
df['motiles'].fillna('OTRA',inplace=True)
df['c_num'].fillna(df.c_num.median(),inplace=True)
df['desc_servicio'].fillna(df.desc_servicio.mode()[0],inplace=True)
df['p_diasgesta'].fillna(df.p_diasgesta.median(),inplace=True)
df['panticoncep'].fillna(df.panticoncep.mode()[0],inplace=True)


# In[604]:


df.isna().sum()


# In[605]:


df.drop(columns=['p_consent', 's_complica', 'c_dolor', 'tanalgesico', 'cconsejo'], inplace = True, axis = 1)


# In[606]:


df.isna().sum()


# In[607]:


df.isna().sum()


# In[608]:


#Normalization - removing redundancies, anomalies, inconsistencies
df


# In[609]:


df = df.drop_duplicates()


# In[610]:


df


# In[ ]:





# In[611]:


df


# In[612]:


#Remove outliers for numeric values
df = df[(np.abs(stats.zscore(df['menarca'])) < 3)]
df = df[(np.abs(stats.zscore(df['fsexual'])) < 3)]
df = df[(np.abs(stats.zscore(df['sememb'])) < 3)]
df = df[(np.abs(stats.zscore(df['p_semgest'])) < 3)]
df = df[(np.abs(stats.zscore(df['p_diasgesta'])) < 3)]
df = df[(np.abs(stats.zscore(df['nhijos'])) < 3)]
df = df[(np.abs(stats.zscore(df['gesta'])) < 3)]
df = df[(np.abs(stats.zscore(df['naborto'])) < 3)]
df = df[(np.abs(stats.zscore(df['npartos'])) < 3)]
df = df[(np.abs(stats.zscore(df['ncesarea'])) < 3)]
df = df[(np.abs(stats.zscore(df['nile'])) < 3)]
df = df[(np.abs(stats.zscore(df['c_num'])) < 3)]


# In[613]:


df


# In[614]:


df.columns


# In[615]:


#Checking for anomalies in columns

df.edocivil_descripcion.unique()


# In[616]:


df.edad.unique()


# In[617]:


df.desc_derechohab.unique()


# In[618]:


df.nivel_edu.unique()


# In[619]:


df.ocupacion.unique()


# In[620]:


df.religion.unique()


# In[621]:


df.parentesco.unique()


# In[622]:


df.menarca.unique()


# In[623]:


df.fsexual.unique()


# In[624]:


df.sememb.unique()


# In[625]:


df.nhijos.unique()


# In[626]:


df.gesta.unique()


# In[627]:


df.naborto.unique()


# In[628]:


df.npartos.unique()


# In[629]:


df.ncesarea.unique()


# In[630]:


df.nile.unique()


# In[631]:


df.consejeria.unique()


# In[632]:


df.anticonceptivo.unique()


# In[633]:


df.c_num.unique()


# In[634]:


df.motiles.unique()


# In[635]:


df.desc_servicio.unique()


# In[636]:


df.p_semgest.unique()


# In[637]:


df.p_diasgesta.unique()


# In[638]:


df.panticoncep.unique()


# In[639]:


df.resultado_ile.unique()


# In[640]:


df.procile_simplificada.unique()


# In[641]:


df.procile.unique()


# In[642]:


print(df.groupby(['ocupacion']).size());


# In[643]:


print(df.groupby(['nivel_edu']).size());


# In[644]:


print(df.groupby(['nhijos']).size())


# In[645]:


df.loc[df['nivel_edu'] == 'DOCTORADO', 'nivel_edu'] = 'POSTGRADO'
df.loc[df['nivel_edu'] == 'MAESTRIA', 'nivel_edu'] = 'POSTGRADO'

df.loc[df['nivel_edu'] == 'SECUNDERIA', 'nivel_edu'] = 'SECUNDARIA'


# In[646]:


df


# In[647]:


sb.set(rc={"figure.figsize":(5, 5)}) 
nabce = sb.lineplot(df['nhijos'], df['naborto'])
nabce.set(xlabel ="Number of Children", ylabel = "Number of Abortions", title ="Number of Abortions by number of children")


# In[648]:


#The number of abortions and the number of children are generally directly proportional to each other. 
#As the number of children initially increases, women are more likely to have more abortions. 
#However, after a woman has 5 children, it seems that the probability that a woman has more abortions does 
#not increase, but only 45 out of the 50,000+ women have more than 4 children and variability is high, so 
#we would need more data to confirm this trend. The probability of abortions increasing between 0 and 1 children could 
#be because women feel that they are too young to have a child. The probability of abortions increasing between 1 and 4 
#children could be because women only want to have a select number of children.


# In[649]:


sb.set(rc={"figure.figsize":(20, 10)}) 
naboc = sb.barplot(df['ocupacion'], df['naborto'])
naboc.set(xlabel ="Occupacion", ylabel = "Number of Abortions", title ="Number of Abortions by Occupacion")


# In[650]:


#The probability of an abortion is lowest if a woman is a student, whereas the probably of an abortion is 
#highest if a woman’s job isn’t listed in the dataset. The probability of abortion is high for domestic unpaid women as well. 
#The probability of an abortion is second lowest if a woman is unemployed. It could be that being a student or being 
#unemployment is associated with a lower probability of an abortion because women may see having children as a key to 
#a stable life.


# In[651]:


nabel = sb.barplot(df['nivel_edu'], df['naborto'])
nabel.set(xlabel ="Education Level", ylabel = "Number of Abortions", title ="Number of Abortions by education level")


# In[652]:


#The probability of an abortion is lowest if a woman is uneducated, whereas the probability 
#of an abortion is highest if a woman has done graduate school. In general, the probability 
#of an abortion is higher for an educated woman. The more educated the woman is, the higher the 
#probability of an abortion. One possible reason for this could be because more educated women want 
#to focus on their career instead of taking pregnancy leave.


# In[653]:


df


# In[ ]:




