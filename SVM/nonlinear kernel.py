#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy.linalg import norm
import pandas as pd
from pprint import pprint
import csv
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der


# In[3]:


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


# In[4]:


print(X.shape)
print(X_test.shape)


# In[5]:


C=np.array([100,500,700])/873


# In[6]:


Gamma = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])


# In[7]:


def kernal(xi,xj,gamma):
    K  = np.exp(-np.transpose(xi-xj)@(xi-xj)/gamma)
    return K 


# In[8]:


def kernal_Hessian(X,y):
    n = len(X)
    H= np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            
            H[i,j] =  y[i]*y[j]*kernal(X[i],X[j],gamma)
    return H


# In[9]:


for c in C:
    print("With the new C")
    for gamma in Gamma:
        
        print("---------------C =", c, " and gamma = ",gamma,"-----------------------")

        H=kernal_Hessian(X,y)

        n_sample, n_dim =X.shape
        N = n_sample
        a= np.zeros(N)

        loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
        I = np.identity(N)
        cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

        a0= np.zeros(N)
        res = minimize(loss, a0, method='SLSQP', constraints=cons)
        a = res.x

        index= []
        for i in range(len(a)):

            if a[i]>0:
                index.append(i)
       
        print(len(index))


        a_supp = a[index]
        y_supp = y[index]
        X_supp = X[index]

        # predict train wphi
        Wphi_train = []

        for tx in range(N):
            train_x = X[tx]
            train_wphi = 0
            for i in range(len(a)):
                xi = X[i]
                train_wphi = train_wphi + a[i]*y[i]*kernal(train_x,xi,gamma)
            
            Wphi_train.append(train_wphi)
            
        # predict test wphi
        Wphi_test = []

        for txx in range(len(X_test)):
            test_x = X_test[txx]
            test_wphi = 0
            for i in range(len(a)):
                xi = X[i]
                test_wphi = test_wphi + a[i]*y[i]*kernal(test_x,xi,gamma)
            
            Wphi_test.append(test_wphi)
        
        # predict b
        b_train =0
        
        for k in range(len(a_supp)):
            yk = y_supp[k]
            xk = X_supp[k]

            sumb = 0
            for i in range(N):
                xi = X[i]
                sumb = sumb +a[i]*y[i]*kernal(xk,xi,gamma)
            b_train =  b_train+(yk -sumb)

        b_train = b_train/len(a_supp)
        
        # Find prediction 
        pred_train =np.sign(Wphi_train +b_train)
        pred_test =np.sign(Wphi_test +b_train)
        
        # compute train and test error
        
        train_error = np.sum(pred_train!= y)/len(X)*100
        test_error = np.sum(pred_test!= y_test)/len(X_test)*100       

        print("The average training error is  ",train_error)
        print("The average testing error is  ",test_error)


        


# In[17]:


c = 500/873

gamma = Gamma[0]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_0= []
for i in range(len(a)):

    if a[i]>0:
        index_0.append(i)
        
print(len(index_0))
#--------------------------
gamma = Gamma[1]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_1= []
for i in range(len(a)):

    if a[i]>0:
        index_1.append(i)

print(len(index_1))
#--------------------------
gamma = Gamma[2]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_2= []
for i in range(len(a)):

    if a[i]>0:
        index_2.append(i)
print(len(index_2))
#----------------------------
gamma = Gamma[3]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_3= []
for i in range(len(a)):

    if a[i]>0:
        index_3.append(i)
print(len(index_3))
#--------------------------
gamma = Gamma[4]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_4= []
for i in range(len(a)):

    if a[i]>0:
        index_4.append(i)

print(len(index_4))
#--------------------------
gamma = Gamma[5]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_5= []
for i in range(len(a)):

    if a[i]>0:
        index_5.append(i)
print(len(index_5))
#---------------------------
gamma = Gamma[6]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_6= []
for i in range(len(a)):

    if a[i]>0:
        index_6.append(i)
print(len(index_6))
#--------------------------
gamma = Gamma[7]
print ("gamma = ", gamma)
H=kernal_Hessian(X,y)

n_sample, n_dim =X.shape
N = n_sample
a= np.zeros(N)

loss = lambda a: np.transpose(a)@H@a/2  - np.sum(a)
I = np.identity(N)
cons = ({'type': 'ineq', 'fun': lambda a: I@a},{'type': 'ineq', 'fun': lambda a: c*np.ones(N) - (I@a) },{'type': 'eq', 'fun': lambda a: np.transpose(a)@y})

a0= np.zeros(N)
res = minimize(loss, a0, method='SLSQP', constraints=cons)
a = res.x

index_7= []
for i in range(len(a)):

    if a[i]>0:
        index_7.append(i)

print(len(index_7))


# In[30]:


def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    
    if (a_set & b_set): 
        print(len(a_set & b_set))
    
    else: 
        print("No common elements")  
    


# In[32]:


common01= common_member(index_0, index_1)
common02= common_member(index_0, index_2)
common03= common_member(index_0, index_3)
common04= common_member(index_0, index_4)
common05= common_member(index_0, index_5)
common06= common_member(index_0, index_6)
common07= common_member(index_0, index_7)

#------------------------------------

common12= common_member(index_1, index_2)
common13= common_member(index_1, index_3)
common14= common_member(index_1, index_4)
common15= common_member(index_1, index_5)
common16= common_member(index_1, index_6)
common17= common_member(index_1, index_7)

#------------------------------------
common23= common_member(index_2, index_3)
common24= common_member(index_2, index_4)
common25= common_member(index_2, index_5)
common26= common_member(index_2, index_6)
common27= common_member(index_2, index_7)

#------------------------------------
common34= common_member(index_3, index_4)
common35= common_member(index_3, index_5)
common36= common_member(index_3, index_6)
common37= common_member(index_3, index_7)

#------------------------------------
common45= common_member(index_4, index_5)
common46= common_member(index_4, index_6)
common47= common_member(index_4, index_7)

#------------------------------------
common56= common_member(index_5, index_6)
common57= common_member(index_5, index_7)

#------------------------------------

common67= common_member(index_6, index_7)


# In[16]:


for gamma in Gamma:
    
    print("------------gamma is ",gamma, "----------------------")
    
    count = np.zeros(N)
    for t in range(10):
        for k in range(N):
            xk = X[k]
            yk = y[k]
            y_pred_k = 0 
            for i in range(N):
                y_pred_k = y_pred_k +count[i]*y[i]*kernal(xk,X[i],gamma)

            #prediction 
            y_pred_k = np.sign(y_pred_k)

            if y_pred_k != yk:
                count[k] = count[k]+1


    # predict train wphi
    pred_train = []

    for tx in range(N):
        train_x = X[tx]
        train_y = 0
        for i in range(N):
            xi = X[i]
            train_y = train_y + count[i]*y[i]*kernal(train_x,X[i],gamma)

        pred_train.append(np.sign(train_y))



    pred_test= []

    for ttx in range(len(X_test)):
        test_x = X_test[ttx]
        test_y = 0
        for i in range(N):
            xi = X[i]
            test_y = test_y + count[i]*y[i]*kernal(test_x,X[i],gamma)

        pred_test.append(np.sign(test_y))

    # compute train and test error

    train_error = np.sum(pred_train!= y)/len(X)*100
    test_error = np.sum(pred_test!= y_test)/len(X_test)*100       

    print("The average training error is  ",train_error)
    print("The average testing error is  ",test_error)






