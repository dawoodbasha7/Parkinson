#!/usr/bin/env python
# coding: utf-8

# # Parkinson disease Predictive model using Machine Learning algorithms

# The data has been downloaded from UCI Machine Learning repository. It is numerical data.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


url="https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"


# In[3]:


data=pd.read_csv(url)
data


# # 0. Atrribute Information
# name - ASCII subject name and recording number\
# MDVP:Fo(Hz) - Average vocal fundamental frequency\
# MDVP:Fhi(Hz) - Maximum vocal fundamental frequency\
# MDVP:Flo(Hz) - Minimum vocal fundamental frequency\
# MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency\
# MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude\
# NHR,HNR - Two measures of ratio of noise to tonal components in the voice\
# status - Health status of the subject (one) - Parkinson's, (zero) - healthy\
# RPDE,D2 - Two nonlinear dynamical complexity measures\
# DFA - Signal fractal scaling exponent\
# spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation\

# # 1. Data Processing

# In[4]:


data.shape


# In[5]:


data.isna().sum()


# The unwanted columns has been removed from the dataset.

# In[6]:


data=data.drop(columns=['name'], axis=1)
data


# ### 1.1 Seperation of dependenet variable and independenet variables

# In[7]:


data_X=data.drop(columns=['status'], axis=1)
data_X


# In[8]:


data_y=data['status']
data_y


# In[9]:


data_y.value_counts()


# ### 1.2 Detecting outliers

# In[10]:


plt.figure(figsize=(10,8))
sns.heatmap(data_X, cmap='Blues')


# ### 1.3 Remove outliers and normalize the data to scale

# In[11]:


import matplotlib.pyplot as plt
data_dist=data.drop(['status'], axis=1)

for i, col in enumerate(data_dist.columns):
    plt.figure(i)
    sns.distplot(data_dist[col])


# We use MinMaxScaler method to scale the features between -1 and 1. Follwing the scaling, fit_transorm method is used to remove outliers if any.

# In[12]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(data_X)
y=data_y


# In[13]:


x_trans=pd.DataFrame(x)
x_trans


# # 2. Model building
# ### 2.1 Support Vector Machine

# In[14]:


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_x,test_x,train_y,test_y=train_test_split(x, y, test_size=0.3, random_state=0)


# In[15]:


plt.scatter(train_x[:,0], train_x[:,1], c=train_y, cmap='winter')


# In[16]:


from sklearn import svm

model = svm.SVC(kernel='linear')
model=model.fit(train_x, train_y)


# In[17]:


prediction_y=model.predict(test_x)
prediction_y, test_y.array


# #### 2.1.1 Accuracy and prediction

# In[18]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(test_y, prediction_y)*100, '%')
print("Precision:",metrics.precision_score(test_y, prediction_y)*100, '%')


# In[19]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, prediction_y)


# In[20]:


from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(test_y,prediction_y, target_names=target_names))


# ### 2.2 Decision Tree

# In[21]:


from sklearn import tree

model_tree=tree.DecisionTreeClassifier()


# In[22]:


decision_tree=model.fit(train_x, train_y)


# In[23]:


predict_decisiontree=decision_tree.predict(test_x)
predict_decisiontree, test_y.array


# #### 2.2.1 Accuracy of Decision Tree

# In[24]:


print("Accuracy:",metrics.accuracy_score(test_y, predict_decisiontree)*100, '%')
print("Precision:",metrics.precision_score(test_y, predict_decisiontree)*100, '%')


# In[25]:


from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, predict_decisiontree)


# ### 2.3 Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


LR=LogisticRegression()
model_LR=LR.fit(train_x, train_y)

prediction_LR=model_LR.predict(test_x)
prediction_LR, test_y.array


# ####  2.3.1 Accuracy test of Logistic Regression

# In[28]:


print("Accuracy:",metrics.accuracy_score(test_y, prediction_LR)*100, '%')
print("Precision:",metrics.precision_score(test_y, prediction_LR)*100, '%')


# In[29]:


confusion_matrix(test_y, prediction_LR)


# # 3. Pipelines

# Instead of going step by step traditional way,by applying pipeline is much easier to avoid monotonous steps. Along with above algorithms, few other classifier algorithms are clubbed in the pipelining process.
# 
# To make the pipeline much lighter, dimensions are being reduced by using PCA. However it has been expecting that some of the data will be loosing. 

# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix


# ### 3.1 Creating the pipeline

# In[31]:


pipe_lr=Pipeline([('scl',StandardScaler()),
                 ('pca',PCA(n_components=7)),
                 ('clf',LogisticRegression(random_state=0))])



pipe_svm=Pipeline([('scl',StandardScaler()),
                  ('pca',PCA(n_components=7)),
                  ('clf',svm.SVC(random_state=0))])




pipe_dt=Pipeline([('scl',StandardScaler()),
                 ('pca',PCA(n_components=7)),
                 ('clf',tree.DecisionTreeClassifier(random_state=0))])

pipe_adaboost=Pipeline([('scl',StandardScaler()),
                       ('pca',PCA(n_components=7)),
                       ('clf',AdaBoostClassifier())])

pipe_gradientboosting=Pipeline([('scl',StandardScaler()),
                       ('pca',PCA(n_components=7)),
                       ('clf',GradientBoostingClassifier(random_state=0))])

pipe_knn=Pipeline([('scl',StandardScaler()),
                  ('pca',PCA(n_components=7)),
                  ('clf',KNeighborsClassifier(n_neighbors=3))])


# #### 3.1.1 List of pipe lines

# In[32]:


pipelines=[pipe_lr,pipe_svm,pipe_dt,pipe_adaboost,pipe_gradientboosting,pipe_knn]


# ### 3.2 Fit the pipelines and accuracy comparision
# 

# During the loop, each classifier has been fitted with training data.

# In[33]:


algos=['Logistic Regression','Support Vector Machine','Decision tree','AdaBoostClassifier','GradientBoosting','KNearestNeighbors']

for (pipe, i) in zip(pipelines,algos):
    pipe.fit(train_x,train_y)
    
    y_pred = pipe.predict(test_x)
    
    print('\033[1m'+'Results of:', i +'\033[0;0m')
    print('pipeline test accuracy: ',round((pipe.score(test_x,test_y)*100),2),'%')
    print('CONFUSION MATRIX: \n',confusion_matrix(test_y, y_pred))
    target_names = ['class 0', 'class 1']
    print(classification_report(test_y,y_pred, target_names=target_names))
    
    if i=='KNearestNeighbors':
        break
    print('---------------------------------------------------------------------')


# ### 3.3 Find the Model with best Accuracy

# In[34]:


pipe_dict={0:'LogisticRegression',1:'Support Vector Machine',2:'Decision tree',3:'AdaBoostClassifier',4:'GradientBoosting',5:'KNearestNeighbors'}

for idx,val in enumerate(pipelines):
    pipe_dict[idx],val.score(test_x,test_y)


# In[35]:


best_acc=0.0
best_clf=0.0
best_pipe=''

for idx, val in enumerate(pipelines):
    if val.score(test_x,test_y)>best_acc:
        best_acc=val.score(test_x,test_y)
        best_pipe=val
        best_clf=idx
    
print('\033[1m'+'Classifier with best accuracy: %s'%pipe_dict[best_clf])


# After using bunch of algorithms, **GradientBoosting** toped the pipeline race. Hence it has been considered that the model associated with **GradientBoosting** is good one.
