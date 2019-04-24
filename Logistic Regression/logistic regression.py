#!/usr/bin/env python
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
Head= ['variance','skewness','curtosis','entropy','label']
data=pd.read_csv('train.csv',names=Head)
test_data=pd.read_csv('test.csv',names= Head)

# replace label "0" to "-1"
data.label = data.label.replace(0,-1)
test_data.label = test_data.label.replace(0,-1)


Feature = ['variance','skewness','curtosis','entropy']
X = data[Feature]
X['aug'] = 1
X = np.array(X)
X_test = test_data[Feature]
X_test['aug'] = 1
X_test = np.array(X_test)

y = data['label']
y = np.array(y)
y_test = test_data['label']
y_test = np.array(y_test)


# In[3]:


C=np.array([0.01, 0.1, 0.5, 1, 3, 5, 10, 100])
print(C)


# In[4]:


def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradL(w,x,y,N,c):
    
    s =-y*np.dot(w,x)
    grad = -N*y*sigmoid(s)*x+w/c
    return grad

def Loss_fun(w,X,y,c):
    loss = np.sum(np.log(1+ np.exp(X@w*y)))+np.dot(w,w)/(2*c)
    return loss

def gradL_like(w,x,y,N):
    
    s =-y*np.dot(w,x)
    grad = -N*y*sigmoid(s)*x
    return grad

def Loss_fun_like(w,X,y):
    loss = np.sum(np.log(1+ np.exp(X@w*y)))
    return loss


# In[5]:


def MAP(c,gamma0,d):
    n_sample, n_dim =X.shape
    N =n_sample
    Loss = []
    w=np.zeros((n_dim,))



    for t in range(100):

        #print("-------------------",t+1,"th epoch------------------------")
        # suffle dataset
        data_suffled=data 
        gamma = gamma0 / (1+gamma0/d*t)
        #print("gamma = ",gamma)
        #data_suffled = data.sample(frac=1).reset_index(drop=True)
        # split suffled data into input X and its output y
        X_suffled = data_suffled[Feature]
        y_suffled = data_suffled['label']

        # Here, x and w should be written in augmented form 
        X_suffled['aug'] = 1
        X_suffled = np.array(X_suffled)
        y_suffled=np.array(y_suffled)

        for i in range(N):

            xi= X_suffled[i,:]
            yi=y_suffled[i]
            
            grad = gradL(w,xi,yi,N,c)
            w = w-gamma*grad
            

        # Evaluate loss function with updated weights   
        loss = Loss_fun(w,X_suffled,y_suffled,c)
        Loss.append(loss)




    #Compute average prediction error on the testdata
    train_error = np.sum(np.sign(X@w)== y)/len(X)*100
    test_error = np.sum(np.sign(X_test@w)== y_test)/len(X_test)*100       

    print("learned weight vector is ", w)
    print("The average testing error is  ",test_error)
    print("The average training error is  ",train_error)


    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(range(len(Loss)),Loss, 'k')
    ax1.set_title('Loss')




    plt.show()




# In[6]:



c=C[0]
gamma0 = 0.001
d = 0.001
MAP(c,gamma0,d)


# In[7]:


c=C[1]
gamma0 = 0.001
d = 0.001
MAP(c,gamma0,d)


# In[8]:


c=C[2]
gamma0 = 0.001
d = 0.001
MAP(c,gamma0,d)


# In[9]:


c=C[3]
gamma0 = 0.001
d = 0.001
MAP(c,gamma0,d)


# In[18]:


c=C[4]
gamma0 = 0.001
d = 0.001
MAP(c,gamma0,d)


# In[33]:


c=C[5]
gamma0 = 0.005
d = 0.005
MAP(c,gamma0,d)


# In[36]:


c=C[6]
gamma0 = 0.05
d = 0.01
MAP(c,gamma0,d)


# In[21]:


c=C[7]
gamma0 = 0.001
d = 0.001
MAP(c,gamma0,d)


# In[44]:


def ML(gamma0,d):
    n_sample, n_dim =X.shape
    N =n_sample
    Loss = []
    w=np.zeros((n_dim,))



    for t in range(100):

        #print("-------------------",t+1,"th epoch------------------------")
        # suffle dataset
        data_suffled=data 
        gamma = gamma0 / (1+gamma0/d*t)
        #print("gamma = ",gamma)
        #data_suffled = data.sample(frac=1).reset_index(drop=True)
        # split suffled data into input X and its output y
        X_suffled = data_suffled[Feature]
        y_suffled = data_suffled['label']

        # Here, x and w should be written in augmented form 
        X_suffled['aug'] = 1
        X_suffled = np.array(X_suffled)
        y_suffled=np.array(y_suffled)

        for i in range(N):

            xi= X_suffled[i,:]
            yi=y_suffled[i]
            
            grad = gradL_like(w,xi,yi,N)
            w = w-gamma*grad
            

        # Evaluate loss function with updated weights   
        loss = Loss_fun_like(w,X_suffled,y_suffled)
        Loss.append(loss)




    #Compute average prediction error on the testdata
    train_error = np.sum(np.sign(X@w)== y)/len(X)*100
    test_error = np.sum(np.sign(X_test@w)== y_test)/len(X_test)*100       

    print("learned weight vector is ", w)
    print("The average testing error is  ",test_error)
    print("The average training error is  ",train_error)


    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(range(len(Loss)),Loss, 'k')
    ax1.set_title('Loss')




    plt.show()




# In[46]:


d = 0.001
gamma0 = 0.001
ML(gamma0,d)


# In[ ]:




