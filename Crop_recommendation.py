#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , cross_validate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , Activation , Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv('Crop_recommendation.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


features = df.T[:7].T
features


# In[6]:


scaler = StandardScaler()
features = scaler.fit_transform(features)
features


# In[7]:


data_classes = list(df['label'].unique())


# In[8]:


data_classes


# In[9]:


targets = df['label'].apply(data_classes.index)
targets


# In[10]:


targets = to_categorical(targets)
targets


# In[11]:


x_train , x_test , y_train  , y_test = train_test_split(features,targets,test_size = 0.1 , random_state = 42)


# In[12]:


x_train.shape , y_train.shape


# In[13]:


model = Sequential()
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(22,activation='softmax'))


# In[14]:


model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[15]:


history = model.fit(x_train,y_train,epochs=50)


# In[20]:


plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')


# In[21]:


plt.plot(history.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuaracy')


# In[18]:


model.evaluate(x_test,y_test)


# In[19]:


predictions = model.predict(x_test)
for i in range(50):
    print('The model predicted the answer is',df['label'].unique()[np.argmax(predictions[i])],'While the real answer is',df['label'].unique()[np.argmax(y_test[i])])


# In[ ]:




