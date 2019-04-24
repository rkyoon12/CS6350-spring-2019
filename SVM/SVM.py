#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from numpy.linalg import norm
import pandas as pd
from pprint import pprint
import csv
import matplotlib.pyplot as plt


# In[13]:


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


# In[14]:


print(X.shape)
print(X_test.shape)


# In[15]:


C=np.array([1,10,50,100,300,500,700])/873
print(C)


# In[16]:


def soft_svm(c,gamma0,d):
    n_sample, n_dim =X.shape
    N =n_sample
    Loss = []
    w=np.zeros(n_dim)



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
            J =0
            nJ =0

            xi= X_suffled[i,:]
            yi=y_suffled[i]
            w0 = w[:-1]

            if yi*w@xi <=1:

                nJ = np.append(w0,[0])-c*N*yi*xi
            else: 
                nJ = np.append(w0,[0])


            w = w-gamma*nJ

        #Ji =  w0@w0/2 + C*N*max(0, 1-y*X_suffled@w)    


        w0 = w[:-1]
        J = 0 
        for i in range(N):
            xi= X_suffled[i,:]
            yi=y_suffled[i]

            J = J + w0@w0/2 + c*N*max(0, 1-yi*xi@w)   
        loss = J/N
        Loss.append(loss)




    #Compute average prediction error on the testdata
    train_error = np.sum(np.sign(X@w)!= y)/len(X)*100
    test_error = np.sum(np.sign(X_test@w)!= y_test)/len(X_test)*100       

    print("learned weight vector is ", w)
    print("The average testing error is  ",test_error)
    print("The average training error is  ",train_error)


    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(range(len(Loss)),Loss, 'k')
    ax1.set_title('Loss')




    plt.show()




# In[18]:


for c in C:
    print("-------when c is ",c )
    #-----------------------
    gamma0 = 0.01
    d =0.01
    print("--------gamma0 =",gamma0, "d = ",d,"---------")
    soft_svm(c,gamma0,d)

    gamma0 = 0.05
    d =0.01
    print("--------gamma0 =",gamma0, "d = ",d,"---------")
    soft_svm(c,gamma0,d)

    gamma0 = 0.01
    d =0.05
    print("--------gamma0 =",gamma0, "d = ",d,"---------")
    soft_svm(c,gamma0,d)

    #-----------------------
    gamma0 = 0.05
    d =0.05
    print("--------gamma0 =",gamma0, "d = ",d,"---------")
    soft_svm(c,gamma0,d)

    


# In[20]:


def soft_svm_b(c,gamma0):
    n_sample, n_dim =X.shape
    N =n_sample
    Loss = []
    w=np.zeros(n_dim)
    for t in range(100):
        #print("-------------------",t+1,"th epoch------------------------")
        # suffle dataset
        data_suffled=data 
        gamma = gamma0 / (1+t)
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
            J =0
            nJ =0

            xi= X_suffled[i,:]
            yi=y_suffled[i]
            w0 = w[:-1]

            if yi*w@xi <=1:

                nJ = np.append(w0,[0])-c*N*yi*xi
            else: 
                nJ = np.append(w0,[0])


            w = w-gamma*nJ

        #Ji =  w0@w0/2 + C*N*max(0, 1-y*X_suffled@w)    


        w0 = w[:-1]
        J = 0 
        for i in range(N):
            xi= X_suffled[i,:]
            yi=y_suffled[i]

            J = J + w0@w0/2 + c*N*max(0, 1-yi*xi@w)   
        loss = J/N
        Loss.append(loss)




    #Compute average prediction error on the testdata
    train_error = np.sum(np.sign(X@w)!= y)/len(X)*100
    test_error = np.sum(np.sign(X_test@w)!= y_test)/len(X_test)*100       

    print("learned weight vector is ", w)
    print("The average testing error is  ",test_error)
    print("The average training error is  ",train_error)


    ax1 = plt.subplot(211)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(range(len(Loss)),Loss, 'k')
    ax1.set_title('Loss')




    plt.show()




# In[21]:


for c in C:
    print("-------when c is ",c )
    
    #-----------------------
    gamma0 = 0.05
   
    print("--------gamma0 =",gamma0, "---------")
    soft_svm_b(c,gamma0)

    ##-----------------------
    gamma0 = 0.01
   
    print("--------gamma0 =",gamma0, "---------")
    soft_svm_b(c,gamma0)


    


# In[ ]:




