
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
data=pd.read_csv('bank-note/train.csv',names=Head)
test_data=pd.read_csv('bank-note/test.csv',names= Head)

Feature = ['variance','skewness','curtosis','entropy']
X = data[Feature]
X['aug'] = 1
X_test = test_data[Feature]
X_test['aug'] = 1

y = data['label']
y_test = test_data['label']


# In[3]:


# replace label "0" to "-1"
data.label = data.label.replace(0,-1)
test_data.label = test_data.label.replace(0,-1)

# learning rate
r =0.1


# In[4]:



n_sample, n_dim =data.shape

w=np.zeros(n_dim)
# inside of for loop





for t in range(10):
    print("-------------------",t+1,"th epoch------------------------")
    # suffle dataset
    data_suffled=data 
    #data_suffled = data.sample(frac=1).reset_index(drop=True)
    # split suffled data into input X and its output y
    X_suffled = data_suffled[Feature]
    y_suffled = data_suffled['label']

    # Here, x and w should be written in augmented form 
    X_suffled['aug'] = 1

    X_suffled=np.array(X_suffled)
    y_suffled=np.array(y_suffled)
    
    

    for i in range(len(X_suffled)):
       

        x_i= X_suffled[i,:]
        y_i=y_suffled[i]


        if y_i*np.dot(x_i,w) <= 0:
              
            w +=r*y_i*x_i
   
   
#Compute average prediction error on the testdata
Error = np.sign(X_test@w)!= y_test
error = np.sum(np.sign(X_test@w)!= y_test)/len(X_test)*100       
   
print("learned weight vector is ", w)
print("The average testing error is  ",error)

     


# In[5]:


#voted perceptron
n_sample, n_dim =data.shape

w=np.zeros(n_dim)
test_Error_vote = []
pred=np.zeros(n_dim)
m=0
# inside of for loop

W=[]
C=[]

c= 1
for t in range(10):
    print("-------------------",t+1,"th epoch------------------------")
    
    data_suffled = data
    
    # split suffled data into input X and its output y
    X_suffled = data_suffled[Feature]
    y_suffled = data_suffled['label']

    # Here, x and w should be written in augmented form 
    X_suffled['aug'] = 1

    X_suffled=np.array(X_suffled)
    y_suffled=np.array(y_suffled)
    
    for i in range(len(X_suffled)):

        x_i= X_suffled[i,:]
        y_i=y_suffled[i]


        if y_i*np.dot(x_i,w) <= 0:
              
            w = w + r*y_i*x_i
            print("count = ",c)
            print("weight = ",w)
            pred = c*np.sign(X_test@w)
           
            W.append(w)
            C.append(c)
            m+=1
            c=1
            prediction = np.sign(pred)
            error = np.sum(prediction!= y_test)/len(X_test)*100
            test_Error_vote.append(error)
            
            
        
        else:
            c+=1
   
            
            
           
   
    

prediction = np.sign(pred)

error = np.sum(prediction!= y_test)/len(X_test)*100
print("------------------------------------------------")
print("The average testing error is  si",error)

     


# In[6]:





ax1 = plt.subplot(211)
ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
ax1.plot(range(len(test_Error_vote)),test_Error_vote, 'k')
ax1.set_title('test_error')


ax2 = plt.subplot(212)
        # Values >0.0 zoom out
ax2.plot(range(len(C)),C,'o-')
ax2.set_title('counts')



plt.show()


# In[7]:


#averaged Perceptron
w=np.zeros(n_dim)
a=np.zeros(n_dim)

for t in range(10):
    # suffle dataset
    data_suffled = data
    #data_suffled = data.sample(frac=1).reset_index(drop=True)
    # split suffled data into input X and its output y
    X_suffled = data_suffled[Feature]
    y_suffled = data_suffled['label']

    # Here, x and w should be written in augmented form 
    X_suffled['aug'] = 1

    X_suffled=np.array(X_suffled)
    y_suffled=np.array(y_suffled)
    
    

    for i in range(len(X_suffled)):

        x_i= X_suffled[i,:]
        y_i=y_suffled[i]
        
        if y_i*np.dot(x_i,w) <= 0:
              
            w =w+r*y_i*x_i
            
            
        a+= w
        
print("The learned weight is ", w)
pred= np.sign(X_test@a)
error = np.sum(pred!= y_test)/len(X_test)*100
print("The average testing error is  ",error)

