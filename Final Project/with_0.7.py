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


Head= ['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA','Research','Chance_of_Admit']
data=pd.read_csv('data/train.csv',names=Head)
test_data=pd.read_csv('data/test.csv',names=Head)
Feature = ['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA','Research']


# In[3]:


def normalization(data,target_head='feature'):
    f_min = np.min(data[target_head])
    f_max = np.max(data[target_head])
    data[target_head] = (data[target_head]-f_min)/(f_max-f_min)


# In[4]:


for f in Feature:
    normalization(data,f)
    normalization(test_data,f)

# change data into array
data = np.array(data)
test_data = np.array(test_data)


# In[5]:


def label2binary(data,threshold):
    n_data = len(data)
    for i in range(n_data):
        if data[i,-1]>= threshold:
            data[i,-1]= 1
        else:
            data[i,-1] = -1


# In[6]:


threshold = 0.70
label2binary(data,threshold)
label2binary(test_data,threshold)


# In[7]:


X_train = data[:,:-1]
y_train = data[:,-1]
X_test = test_data[:,:-1]
y_test = test_data[:,-1] 


# In[8]:


n_pre_train , n_pre_dim = np.shape(X_train)
n_pre_test, n_pre_dim = np.shape(X_test)

Augmented_train = np.ones((n_pre_train,n_pre_dim+1))
Augmented_train[:,:-1] = X_train
X_train = Augmented_train

Augmented_test = np.ones((n_pre_test,n_pre_dim+1))
Augmented_test[:,:-1] = X_test
X_test = Augmented_test


# In[9]:


n_train, n_dim = np.shape(X_train)
n_test = len(y_test)
print(n_train,n_dim)


# In[10]:


def average_perceptron(X_train, y_train,w,a):
    n_train = len(X_train)
    for i in range(n_train):
        xi = X_train[i,:]
        yi = y_train[i]

        if yi*np.dot(xi,w) <= 0: # for misclassified data
            w = w + r*yi*xi
        
        a = a+w
    return w,a


# In[11]:


R = [1, 0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
w=np.zeros(n_dim)
a=np.zeros(n_dim)
Accuracy_train = []
Accuracy_test=[]
for r in R:
    
    for epoch in range(10):
        w,a = average_perceptron(X_train, y_train,w,a)
    
    pred_train = np.sign(X_train@a)
    pred_test = np.sign(X_test@a)

    accuracy_train = np.sum(pred_train == y_train)/n_train*100
    accuracy_test = np.sum(pred_test == y_test)/n_test*100
    
    Accuracy_train.append(accuracy_train)
    Accuracy_test.append(accuracy_test)


# In[12]:


plt.plot(range(len(R)),Accuracy_train,'b*-')
plt.plot(range(len(R)),Accuracy_test,'g*-')


# In[13]:


def soft_SVM(X_train,y_train, w,learning_rate, c):
    n_train = len(X_train)
    loss = 0
    for i in range(n_train):
        dloss  = 0

        xi = X_train[i,:]
        yi = y_train[i]
        w0 = w[:-1]

        if yi*np.dot(w,xi) <= 1: # misclassified or inside the margin
            dloss = np.append(w0,[0])-c*n_train*yi*xi
        else:
            dloss = np.append(w0,[0])
        
        w = w-learning_rate*dloss
        
        w0 = w[:-1]
        loss = loss+ np.dot(w0,w0)/2 + c*n_train*max(0,1-yi*np.dot(xi,w))
    
    return w,loss/n_train


# In[14]:


Loss = []
w = np.zeros(n_dim)
learning_rate = 0.0001
c = 1# hyperparameters
for epoch in range(200):
    w,loss = soft_SVM(X_train,y_train, w,learning_rate, c) 
    Loss.append(loss)

pred_train = np.sign(X_train@w)
pred_test = np.sign(X_test@w)

accuracy_train = np.sum(pred_train == y_train)/n_train*100
accuracy_test = np.sum(pred_test == y_test)/n_test*100

print("Train accuracy = ",accuracy_train)
print("Test accuracy = ",accuracy_test)



ax1 = plt.subplot(211)
ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
ax1.plot(range(len(Loss)),Loss, 'k')
ax1.set_title('Loss')


# In[15]:


def kernal(xi,xj,gamma):
    K  = np.exp(-np.transpose(xi-xj)@(xi-xj)/gamma)
    return K 
def kernal_Hessian(X,y,gamma):
    n = len(X)
    H= np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i,j] =  y[i]*y[j]*kernal(X[i],X[j],gamma)
    return H


# In[16]:


#c= 7/8
#learning_rate = 0.5
c=1
learning_rate = 0.5


# In[17]:


H = kernal_Hessian(X_train, y_train,learning_rate)
a= np.zeros(n_train)


# In[18]:


loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(n_train)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(n_train) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y_train})

a0= np.zeros(n_train)
res = minimize(loss, a0, method='SLSQP', constraints=cons)

# search optimal alpha
a = res.x

#find index for support vectors
index= []
for i in range(n_train):

    if a[i]>0:
        index.append(i)

a_supp = a[index]
y_train_supp = y_train[index]
X_train_supp = X_train[index]


# In[19]:


# predict train wphi
Wphi_train = []

for tx in range(n_train):
    train_x = X_train[tx]
    train_wphi = 0
    for i in range(n_train):
        xi = X_train[i]
        train_wphi = train_wphi + a[i]*y_train[i]*kernal(train_x,xi,learning_rate)

    Wphi_train.append(train_wphi)

# predict test wphi
Wphi_test = []

for txx in range(n_test):
    test_x = X_test[txx]
    test_wphi = 0
    for i in range(n_train):
        xi = X_train[i]
        test_wphi = test_wphi + a[i]*y_train[i]*kernal(test_x,xi,learning_rate)

    Wphi_test.append(test_wphi)

# predict b
b_train =0

for k in range(len(a_supp)):
    yk = y_train_supp[k]
    xk = X_train_supp[k]

    sumb = 0
    for i in range(n_train):
        xi = X_train[i]
        sumb = sumb +a[i]*y_train[i]*kernal(xk,xi,learning_rate)
    b_train =  b_train+(yk -sumb)

b_train = b_train/len(a_supp)



# In[20]:


# Find prediction 
pred_train =np.sign(Wphi_train +b_train)
pred_test =np.sign(Wphi_test +b_train)

# compute train and test error

accuracy_train = np.sum(pred_train == y_train)/n_train*100
accuracy_test = np.sum(pred_test == y_test)/n_test*100       

print("Train accuracy = ",accuracy_train)
print("Test accuracy = ",accuracy_test)


# In[21]:


def sigmoid(s):
        # activation function 
    return 1/(1+np.exp(-s))
def sigmoidPrime(s):
        #derivative of sigmoid
    return s * (1 - s)



# In[22]:


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


# In[23]:


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


# In[24]:


def train_NN_random(n_h1,gamma0,d):
    n_sample, n_dim =X_train.shape
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
       

        # learning rate
        gamma = gamma0 / (1+gamma0/d*t)

        for i in range(n_sample):
            xi= np.reshape(X_train[i,:],(1,n_dim))
            yi=y_train[i]

            cache = forward_prop(W1,b1,W2,b2,W3,b3,xi)
            grads = backward_prop(W1,b1,W2,b2,W3,b3,cache,yi)
            dW3,db3,dW2,db2,dW1,db1 = grads['dW3'],grads['db3'],grads['dW2'],grads['db2'],grads['dW1'],grads['db1']

            W1 = W1 - gamma*dW1
            W2 = W2 - gamma*dW2
            W3 = W3 - gamma*dW3
            b1 = b1 - gamma*db1
            b2 = b2 - gamma*db2
            b3 = b3 - gamma*db3

        updated_cache = forward_prop(W1,b1,W2,b2,W3,b3,X_train)
        updated_output = updated_cache['a3'].reshape(n_train,)
        loss = np.sum((updated_output-y_train)**2)/2
        Loss.append(loss)
    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(range(len(Loss)),Loss, 'k')
    ax1.set_title('Loss')




    plt.show()

    train_cache = forward_prop(W1,b1,W2,b2,W3,b3,X_train)
    train_output =train_cache['a3']

    test_cache = forward_prop(W1,b1,W2,b2,W3,b3,X_test)
    test_output =test_cache['a3']

    train_acc = np.sum((np.sign(train_output).reshape((n_train,))==y_train)) /n_sample*100
    print("train accuracy is",train_acc )
    test_acc = np.sum((np.sign(test_output).reshape((len(y_test),))==y_test) )/len(y_test)*100
    print( "test accuracy is",test_acc)




# In[25]:


train_NN_random(5,0.1,0.1)


# In[26]:


train_NN_random(10,1,1)


# In[27]:


train_NN_random(25,0.1,0.1)


# In[ ]:




