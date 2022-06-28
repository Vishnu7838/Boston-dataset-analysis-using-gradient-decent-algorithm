#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


from sklearn import preprocessing


# In[6]:


data = pd.read_csv("train_x_y.csv", delimiter=",")
data.shape


# In[9]:


data.columns


# In[15]:


data.head()


# In[ ]:





# In[19]:


data=preprocessing.scale(data)
data=pd.DataFrame(data)
data.head()


# In[21]:


lc=np.ones(data.shape[0])
lc.shape
data['lc']=lc


# In[23]:


points=np.array(data)
points.shape


# In[26]:


points.shape


# In[28]:


def step_gradient(points, learning_rate, m):
    m_slope = np.zeros(14)
    M = len(points)
    for i in range(M):
        x = points[i]
        y = Y[i]
        for j in range(14):
            m_slope[j] = m_slope[j]+(-2/M)* (y - (m * x).sum() )*x[j]
            
    for j in range(14):
        m[j] = m[j] - learning_rate*m_slope[j]
    
    return m


# In[37]:


def gd(points, learning_rate, num_iterations):
    m=np.zeros(14)
    for i in range(num_iterations):
        m = step_gradient(points, learning_rate, m)
        print(i, "Cost: ", cost(points, m))
    return m


# In[32]:


def cost(points, m):
    total_cost = 0
    M = len(points)
    for i in range(M):
        x = points[i]
        y = Y[i]
        total_cost += (1/M)*((y - (m*x).sum() )**2)
    return total_cost


# In[34]:


def run():
    learning_rate = 0.07
    num_iterations = 900
    m = gd(points, learning_rate, num_iterations)
    print(m)
    return m


# In[38]:


m=run()


# In[40]:


m


# In[48]:


df=pd.read_csv("test_x.csv")

df[1]=np.ones(df.shape[0])


# In[50]:


y_test=[]
x1=np.array(df)
M = x1.shape[0]
for i in range(M):
    x = x1[i]
    z=(m*x).sum()
    y_test.append(z)
    


# In[52]:


add_row=df.columns
add_row
row=[]
for i in add_row:
    row.append(float(i))
row=np.array(row)
z=(row*m).sum()
z


# In[54]:


y_test.insert(0,z)


# In[58]:


np.savetxt('bostontestgd0.07.csv', y_test, delimiter=',',fmt='%1.5f')


# In[60]:


dataframe=np.genfromtxt('test_x.csv',delimiter=",")
dataframe.shape
ras=np.array(dataframe)
print(ras.shape)
ras=np.insert(ras,ras.shape[1],np.ones(ras.shape[0]),axis=1)
ras.shape


# In[62]:


y_test=[]
M = ras.shape[0]
for i in range(M):
    x = ras[i]
    z=(m*x).sum()
    y_test.append(z)
np.savetxt('bostontestgd_copy.csv', y_test, delimiter=',',fmt='%1.5f')


# In[63]:


# scaling
std_scale = preprocessing.StandardScaler().fit(points)
points= std_scale.transform(points)

