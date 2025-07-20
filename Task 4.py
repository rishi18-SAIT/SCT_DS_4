#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("RTA Dataset.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.describe(include="all")


# In[6]:


df.info()


# In[7]:


df.duplicated().sum()


# In[8]:


df['Accident_severity'].value_counts()


# In[9]:


#plotting the final class
sns.countplot(x = df['Accident_severity'])
plt.title('Distribution of Accident severity')


# In[10]:


df.isna().sum()


# In[11]:


#dropping columns which has more than 2500 missing values and Time column
df.drop(['Service_year_of_vehicle','Defect_of_vehicle','Work_of_casuality', 'Fitness_of_casuality','Time'],
        axis = 1, inplace = True)
df.head()


# In[12]:



#storing categorical column names to a new variable
categorical=[i for i in df.columns if df[i].dtype=='O']
print('The categorical variables are',categorical)


# In[13]:



#for categorical values we can replace the null values with the Mode of it
for i in categorical:
    df[i].fillna(df[i].mode()[0],inplace=True)


# In[14]:



#checking the current null values
df.isna().sum()


# # DATA VISUALIZATION

# In[15]:


#plotting relationship between Number_of_casualties and Number_of_vehicles_involved
sns.scatterplot(x=df['Number_of_casualties'], y=df['Number_of_vehicles_involved'], hue=df['Accident_severity'])


# In[16]:



#joint Plot
sns.jointplot(x='Number_of_casualties',y='Number_of_vehicles_involved',data=df)


# In[17]:


#checking the correlation betweenn numerical columns
df.corr()


# In[18]:


#plotting the correlation using heatmap
sns.heatmap(df.corr())


# In[19]:


#storing numerical column names to a variable
numerical=[i for i in df.columns if df[i].dtype!='O']
print('The numerica variables are',numerical)


# In[20]:


#distribution for numerical columns
plt.figure(figsize=(10,10))
plotnumber = 1
for i in numerical:
    if plotnumber <= df.shape[1]:
        ax1 = plt.subplot(2,2,plotnumber)
        plt.hist(df[i],color='red')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('frequency of '+i, fontsize=10)
    plotnumber +=1


# In[21]:


#count plot for categorical values
plt.figure(figsize=(10,200))
plotnumber = 1

for col in categorical:
    if plotnumber <= df.shape[1] and col!='Pedestrian_movement':
        ax1 = plt.subplot(28,1,plotnumber)
        sns.countplot(data=df, y=col, palette='muted')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(col.title(), fontsize=14)
        plt.xlabel('')
        plt.ylabel('')
    plotnumber +=1


# # Handling Categorical Values

# In[22]:


df.dtypes


# In[23]:


#importing label encoing module
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#creating a new data frame from performing the chi2 analysis
df1=pd.DataFrame()

#adding all the categorical columns except the output to new data frame
for i in categorical:
    if i!= 'Accident_severity':
        df1[i]=le.fit_transform(df[i])


# In[24]:


#confirming the data type
df1.info()


# In[25]:


plt.figure(figsize=(22,17))
sns.set(font_scale=1)
sns.heatmap(df1.corr(), annot=True)


# In[26]:


#label encoded data set
df1.head()


# In[27]:



#import chi2 test
from sklearn.feature_selection import chi2
f_p_values=chi2(df1,df['Accident_severity'])


# In[28]:



#f_p_values will return Fscore and pvalues
f_p_values


# In[29]:


#for better understanding and ease of access adding them to a new dataframe
f_p_values1=pd.DataFrame({'features':df1.columns, 'Fscore': f_p_values[0], 'Pvalues':f_p_values[1]})
f_p_values1


# In[30]:



#since we want lower Pvalues we are sorting the features
f_p_values1.sort_values(by='Pvalues',ascending=True)


# In[31]:


#after evaluating we are removing lesser important columns and storing to a new data frame
df2=df.drop(['Owner_of_vehicle', 'Type_of_vehicle', 'Road_surface_conditions', 'Pedestrian_movement',
         'Casualty_severity','Educational_level','Day_of_week','Sex_of_driver','Road_allignment',
         'Sex_of_casualty'],axis=1)
df2.head()


# In[32]:


df2.shape


# In[33]:


df2.info()


# In[34]:


#to check distinct values in each categorical columns we are storing them to a new variable
categorical_new=[i for i in df2.columns if df2[i].dtype=='O']
print(categorical_new)


# In[35]:



for i in categorical_new:
    print(df2[i].value_counts())


# In[36]:



#get_dummies
dummy=pd.get_dummies(df2[['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience',
                          'Area_accident_occured', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                          'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                          'Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident']],drop_first=True)
dummy.head()


# In[37]:


#concatinate dummy and old data frame
df3=pd.concat([df2,dummy],axis=1)
df3.head()


# In[38]:


#dropping dummied columns
df3.drop(['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience', 'Area_accident_occured', 'Lanes_or_Medians',
          'Types_of_Junction', 'Road_surface_type', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
          'Vehicle_movement','Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident'],axis=1,inplace=True)
df3.head()


# # Sepearting Independent and Dependent

# In[39]:


x=df3.drop(['Accident_severity'],axis=1)
x.shape


# In[40]:


x.head()


# In[41]:



y=df3.iloc[:,2]
y.head()


# In[42]:


#checking the count of each item in the output column
y.value_counts()


# In[43]:



#plotting count plot using seaborn
sns.countplot(x = y, palette='muted')


# In[44]:


get_ipython().system('pip install imbalanced-learn')


# In[45]:


get_ipython().system('pip install imbalanced-learn')


# In[46]:


import sys
print(sys.executable)


# In[47]:


get_ipython().system('C:\\ProgramData\\Anaconda3\\python.exe -m pip install imbalanced-learn')


# In[48]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE()
x_o, y_o = oversample.fit_resample(x, y)


# In[49]:



#checking the oversampling output
y1=pd.DataFrame(y_o)
y1.value_counts()


# In[50]:


sns.countplot(x = y_o, palette='muted')


#  # Splitting the data

# In[51]:


#converting data to training data and testing data
from sklearn.model_selection import train_test_split
#splitting 70% of the data to training data and 30% of data to testing data
x_train,x_test,y_train,y_test=train_test_split(x_o,y_o,test_size=0.30,random_state=42)


# In[52]:


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# # KNN Model Creation

# # Prediction

# In[53]:



#KNN model alg
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(x_train,y_train)


# In[54]:


y_pred=model_KNN.predict(x_test)


# In[55]:


y_pred


# # Checking Accuracy, Classification Report, Confusion Matrix

# In[56]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay


# # CLASSIFICATION REPORT

# In[57]:


report_KNN=classification_report(y_test,y_pred)
print(report_KNN)


# In[58]:


accuracy_KNN=accuracy_score(y_test,y_pred)
print(accuracy_KNN)


# # Confusion Matrix

# In[59]:


matrix_KNN=confusion_matrix(y_test,y_pred)
print(matrix_KNN,'\n')
print(ConfusionMatrixDisplay.from_predictions(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




