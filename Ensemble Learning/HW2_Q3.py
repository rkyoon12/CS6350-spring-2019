
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import norm
import pandas as pd
from pprint import pprint
import csv
import matplotlib.pyplot as plt


# In[2]:


# import csv file using panda
Head= ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','output']
data=pd.read_csv('train.csv',names=Head)
test_data=pd.read_csv('test.csv',names= Head)

Feature = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr']
X= data[Feature]
X_test= test_data[Feature]

Y = data['output']
Y_test = test_data['output']


# In[3]:


n_data, dim= X.shape
w = np.ones([dim,]) 
x=np.array(X)
y=np.array(Y)


# In[4]:


def cost(w,X,Y):
    J= 0 
    n=len(X)
    
    pred = np.matmul(X,w)
   
    J = (1/2)*np.sum(np.square(Y-pred))
    
    return J


# In[5]:


def grad_cost(w,X,Y):
    n_data, dim = X.shape
       
    pred = Y-np.matmul(X,w)
    dJ = -np.matmul(X.transpose(),pred)
    
    return dJ
        


# In[6]:


def Error(v):
    err = 0
    for i in v:
        err += i**2
    return np.sqrt(err)
        
        


# In[7]:


# define the cost function 
w = np.zeros([dim,]) 
cost_his = []
r=1/128
tol = 1e-6
n_iter = 0
error = 100 
while error > tol:
    J = cost(w,X,Y)
    dJ = grad_cost(w,X,Y)
    
    error = Error(r*dJ)
    #update
    w = w - r*dJ
    n_iter +=1
    #print("%d th iteration's cost is "% n_iter,  J)
    cost_his.append(cost(w,x,y))


# In[8]:


print("-----------At %d-th iteration, this algorithm converges.--------------------" % n_iter)
print("error is ", error)
print("cost function is ", J)
print("its wegith is  ", w)
plt.plot(range(n_iter),cost_his,'b.')


# In[16]:


# define the cost function 
for i in range(8):
    w = np.zeros([dim,]) 
    cost_his = []
    r=1/1024
    tol = 1e-6
    n_iter = 0
    error = 100 
    x=np.array(X)
    y=np.array(Y)
    J=0
    while error > tol:

        rand_ind = np.random.randint(0,n_data)
        x_i = x[rand_ind,:]
        y_i = y[rand_ind]


        w = w +r*(y_i-np.inner(w,x_i))*x_i
        error = Error(r*(y_i-np.inner(w,x_i))*x_i)
        J=cost(w,x,y)

        n_iter +=1
        cost_his.append(J)
        #print(n_iter,error, cost(w,x,y))    


# In[17]:


print("-----------At %d-th iteration, this algorithm converges.--------------------" % n_iter)
print("error is ", error)
print("cost function is ", J)
print("its wegith is  ", w)
plt.plot(range(n_iter),cost_his,'b.')


# In[19]:


ideal_w = np.linalg.pinv(X@X.transpose())@X@Y
print("optimal weight is", ideal_w)
print("with this weight, cost function value is " ,cost(ideal_w,X,Y))

