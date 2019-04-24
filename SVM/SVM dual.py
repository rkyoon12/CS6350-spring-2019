#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import norm
import pandas as pd
from pprint import pprint
import csv
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der


# In[2]:


# import csv file using panda
Head= ['variance','skewness','curtosis','entropy','label']
data=pd.read_csv('train.csv',names=Head)
test_data=pd.read_csv('test.csv',names= Head)

# replace label "0" to "-1"
data.label = data.label.replace(0,-1)
test_data.label = test_data.label.replace(0,-1)


Feature = ['variance','skewness','curtosis','entropy']
X = data[Feature]
#X['aug'] = 1
X = np.array(X)
X_test = test_data[Feature]
#X_test['aug'] = 1
X_test = np.array(X_test)

y = data['label']
y = np.array(y)
y_test = test_data['label']
y_test = np.array(y_test)


# In[3]:


print(X.shape)
print(X_test.shape)


# In[4]:


C=np.array([100,500,700])/873
print(C)


# In[5]:


def Hessian(X,y):
    n = len(X)
    H= np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i,j] =  y[i]*y[j]*X[i]@X[j]
    return H

        


# In[12]:


for c in C:
    print(" ------------when c is ", c,"------------------")
    H=Hessian(X,y)

    n_sample, n_dim =X.shape
    N = n_sample

    loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
    I = np.identity(N)
    cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) -(I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

    a0= np.zeros(N)
    res = minimize(loss, a0, method='SLSQP', constraints=cons, options ={'ftol':1e-15} )
    a = res.x
    
    # find index of support function 
    index = []
    
    for i in range(len(a)):
        if a[i]>0:
            index.append(i)
            

    a_supp = a[index]
    y_supp = y[index]
    X_supp = X[index]
    
    w0 = np.transpose(a_supp*y_supp)@X_supp
    b = np.sum(y_supp - X_supp@w0)/len(X_supp)
    w = np.append(w0,b)
    

    train_error = np.sum(np.sign(X@w0+b)!= y)/len(X)*100
    test_error = np.sum(np.sign(X_test@w0+b)!= y_test)/len(X_test)*100       

    print("learned weight vector is ", w)
    print("The average testing error is  ",test_error)
    print("The average training error is  ",train_error)

