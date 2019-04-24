#!/usr/bin/env python
# coding: utf-8

# In[37]:


from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn


# In[38]:


# import csv file using panda
Head= ['variance','skewness','curtosis','entropy','label']
data=pd.read_csv('train.csv',names=Head)
test_data=pd.read_csv('test.csv',names= Head)

# replace label "0" to "-1"
data.label = data.label.replace(0,-1)
test_data.label = test_data.label.replace(0,-1)


Feature = ['variance','skewness','curtosis','entropy']
X = data[Feature]

X = np.array(X)
X_test = test_data[Feature]

X_test = np.array(X_test)

y = data['label']
y = np.array(y)
y_test = test_data['label']
y_test = np.array(y_test)


# In[39]:


def sigmoid(s):
        # activation function 
    return 1/(1+np.exp(-s))
def sigmoidPrime(s):
        #derivative of sigmoid
    return s * (1 - s)


# In[40]:


def forward_prop(W1,b1,W2,b2,W3,b3,a0):
    
    # Do the first Linear step 
    z1 = a0.dot(W1) + b1
    
    # Put it through the first activation function
    a1 = sigmoid(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Put through second activation function
    a2 = sigmoid(z2)
    
    #Third linear step
    z3 = a2.dot(W3) + b3
    
    #For the Third linear activation function we use the softmax function
    a3 = z3
    
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'a3':a3,'z3':z3}
    return cache


# In[41]:


# This is the backward propagation function
def backward_prop(W1,b1,W2,b2,W3,b3,cache,y):

    # Load forward propagation results
    a0,a1, a2,a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
   
    # Calculate loss derivative with respect to output
    dz3 = a3-y
    
    # Calculate loss derivative with respect to second layer weights
    dW3 = (a2.T).dot(dz3) #dW2 = 1/m*(a1.T).dot(dz2) 
    
    # Calculate loss derivative with respect to second layer bias
    db3 = dz3
    
    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T) ,sigmoidPrime(a2))
    # Calculate loss derivative with respect to first layer weights
    dW2 = np.dot(a1.T, dz2)
    
    # Calculate loss derivative with respect to first layer bias
    db2 = dz2
    
    
    
    dz1 = np.multiply(dz2.dot(W2.T),sigmoidPrime(a1))
    
    dW1 = np.dot(a0.T,dz1)
    
    db1 = dz1
    
    # Store gradients
    grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads


# In[25]:


def train_NN_random(n_h1,gamma0,d):
    n_sample, n_dim =X.shape
    n_h2 = n_h1
    Loss =[]

    # initialize
    W1 = np.random.randn(n_dim,n_h1-1)
    b1 = np.random.randn(1,n_h1-1)
    W2 = np.random.randn(n_h1-1,n_h2-1)
    b2 = np.random.randn(1,n_h2-1)
    W3 = np.random.randn(n_h2-1,1)
    b3 = np.random.randn(1,)


    #for t 
    for t in range(100):
        #suffled data
        data_suffled=data 

        #print("gamma = ",gamma)
        #data_suffled = data.sample(frac=1).reset_index(drop=True)
        # split suffled data into input X and its output y
        X_suffled = data_suffled[Feature]
        y_suffled = data_suffled['label']

        # Here, x and w should be written in augmented form 

        X_suffled = np.array(X_suffled)
        y_suffled=np.array(y_suffled)

        # learning rate
        gamma = gamma0 / (1+gamma0/d*t)

        for i in range(n_sample):
            xi= np.reshape(X_suffled[i,:],(1,4))
            yi=y_suffled[i]

            cache = forward_prop(W1,b1,W2,b2,W3,b3,xi)
            grads = backward_prop(W1,b1,W2,b2,W3,b3,cache,yi)
            dW3,db3,dW2,db2,dW1,db1 = grads['dW3'],grads['db3'],grads['dW2'],grads['db2'],grads['dW1'],grads['db1']

            W1 = W1 - gamma*dW1
            W2 = W2 - gamma*dW2
            W3 = W3 - gamma*dW3
            b1 = b1 - gamma*db1
            b2 = b2 - gamma*db2
            b3 = b3 - gamma*db3

        updated_cache = forward_prop(W1,b1,W2,b2,W3,b3,X)
        updated_output = updated_cache['a3'].reshape(872,)
        loss = np.sum((updated_output-y)**2)/2
        Loss.append(loss)
    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(range(len(Loss)),Loss, 'k')
    ax1.set_title('Loss')




    plt.show()

    train_cache = forward_prop(W1,b1,W2,b2,W3,b3,X)
    train_output =train_cache['a3']

    test_cache = forward_prop(W1,b1,W2,b2,W3,b3,X_test)
    test_output =test_cache['a3']

    train_error = np.sum((np.sign(train_output).reshape((872,))==y) )/n_sample*100
    print("train accuracy is",train_error )
    test_error = np.sum((np.sign(test_output).reshape((len(y_test),))==y_test) )/len(y_test)*100
    print( "test accuracy is",test_error )



# In[48]:


train_NN_random(5,0.1,0.1)


# In[49]:


train_NN_random(10,0.1,0.1)


# In[50]:


train_NN_random(25,0.1,0.1)


# In[52]:


train_NN_random(50,0.1,0.1)


# In[53]:


train_NN_random(100,0.1,0.1)


# In[54]:


def train_NN_zero(n_h1, gamma0, d):
    n_sample, n_dim =X.shape
    n_h2 = n_h1
    Loss =[]

    # initialize
    W1 = np.zeros((n_dim,n_h1-1))
    b1 = np.zeros((1,n_h1-1))
    W2 = np.zeros((n_h1-1,n_h2-1))
    b2 = np.zeros((1,n_h2-1))
    W3 = np.zeros((n_h2-1,1))
    b3 = np.zeros((1,))


    #for t 
    for t in range(100):
        #suffled data
        data_suffled=data 

        #print("gamma = ",gamma)
        #data_suffled = data.sample(frac=1).reset_index(drop=True)
        # split suffled data into input X and its output y
        X_suffled = data_suffled[Feature]
        y_suffled = data_suffled['label']

        # Here, x and w should be written in augmented form 

        X_suffled = np.array(X_suffled)
        y_suffled=np.array(y_suffled)

        # learning rate
        gamma = gamma0 / (1+gamma0/d*t)

        for i in range(n_sample):
            xi= np.reshape(X_suffled[i,:],(1,4))
            yi=y_suffled[i]

            cache = forward_prop(W1,b1,W2,b2,W3,b3,xi)
            grads = backward_prop(W1,b1,W2,b2,W3,b3,cache,yi)
            dW3,db3,dW2,db2,dW1,db1 = grads['dW3'],grads['db3'],grads['dW2'],grads['db2'],grads['dW1'],grads['db1']

            W1 = W1 - gamma*dW1
            W2 = W2 - gamma*dW2
            W3 = W3 - gamma*dW3
            b1 = b1 - gamma*db1
            b2 = b2 - gamma*db2
            b3 = b3 - gamma*db3

        updated_cache = forward_prop(W1,b1,W2,b2,W3,b3,X)
        updated_output = updated_cache['a3'].reshape(872,)
        loss = np.sum((updated_output-y)**2)/2
        Loss.append(loss)
    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(range(len(Loss)),Loss, 'k')
    ax1.set_title('Loss')




    plt.show()

    train_cache = forward_prop(W1,b1,W2,b2,W3,b3,X)
    train_output =train_cache['a3']

    test_cache = forward_prop(W1,b1,W2,b2,W3,b3,X_test)
    test_output =test_cache['a3']

    train_error = np.sum((np.sign(train_output).reshape((872,))==y) )/n_sample*100
    print("train accuracy is",train_error )
    test_error = np.sum((np.sign(test_output).reshape((len(y_test),))==y_test) )/len(y_test)*100
    print( "test accuracy is",test_error )




# In[55]:


train_NN_zero(5,0.1,0.1)


# In[56]:


train_NN_zero(10,0.1,0.1)


# In[57]:


train_NN_zero(25,0.1,0.1)


# In[58]:


train_NN_zero(50,0.1,0.1)


# In[59]:


train_NN_zero(100,0.1,0.1)


# In[ ]:




