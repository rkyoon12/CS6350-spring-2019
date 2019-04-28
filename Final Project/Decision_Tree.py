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
Attributes = Feature


# In[3]:


def normalization(data,target_head='feature'):
    f_min = np.min(data[target_head])
    f_max = np.max(data[target_head])
    data[target_head] = (data[target_head]-f_min)/(f_max-f_min)


# In[4]:


def num2bi(data,test_data,f='feature'):
    med_d = data[f].median()
    min_d = data[f].min()
    max_d = data[f].max()
    bins = [min_d-1,med_d,max_d]
    data[f]=pd.cut(data[f],bins,labels=[0,1])
    test_data[f]=pd.cut(test_data[f],bins,labels=[0,1])


# In[5]:


for f in Feature:
    
    normalization(data,f)
    normalization(test_data,f)
    if f != 'Research':
        num2bi(data,test_data,f)
   
    


# In[7]:


# Change threshold here.
threshold = 0.8
a = data['Chance_of_Admit']>= threshold
label_train = []
for i in range(len(a)):
    if a[i]== True:
        label_train.append(1)
    else:
        label_train.append(0)
b = test_data['Chance_of_Admit']>= threshold
label_test = []
for i in range(len(b)):
    if b[i]== True:
        label_test.append(1)
    else:
        label_test.append(0)


# In[8]:


data.Chance_of_Admit=label_train
test_data.Chance_of_Admit=label_test


# In[9]:


y_test= test_data['Chance_of_Admit']
y_train = data['Chance_of_Admit']


# In[10]:


# Define gain methods
def entropy(target_col):
    elements, counts = np.unique(target_col,return_counts = True)
    entropy = 0
    
    for i in range(len(elements)):    
        p_i = counts[i]/np.sum(counts)
        entropy += -p_i*np.log2(p_i)    
    
    return entropy 


# In[11]:


def InfoGain_base(base,data,attribute,label_name='class'):
    """1 : entropy, 2 : majority error, 3 : Gini index"""
    if base ==1:
        total_entropy = entropy(data[label_name])
        values, counts = np.unique(data[attribute],return_counts=True)

        weighted_entropy = 0 
        for i in range(len(values)):
            sub_data= data[data[attribute]==values[i]]
            weighted_entropy += counts[i]/np.sum(counts)*entropy(sub_data[label_name])

        InfoGain = total_entropy - weighted_entropy

        return InfoGain
    
    
# Find the common label which will be the leaf node.  
def Common_label(data, label='class'):
    
    feature, f_count = np.unique(data[label],return_counts= True)
    common_label = feature[np.argmax(f_count)]
    
    return common_label
    


# In[12]:


def ID3_depth_entropy(depth, data,Attributes,label='class'):
    common_label=Common_label(data,label)
    

    # if all label_values have same value, return itself
    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
        #return 

    # if feature space is empty, return the common label value
    elif len(Attributes)==0:
        return common_label

    # go to the scheme
    else:
       # select the attribute which best split the dataset using InfoGain
        for f in Attributes:
            item_values=[ InfoGain_base(1,data,f,label) for f in Attributes]

        best_attribute_index = np.argmax(item_values)
        best_attribute = Attributes[best_attribute_index]
        
        tree = {best_attribute:{}}

        
        # grow a branch under the root node
        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute]== value]
            sub_common_label= Common_label(sub_data,label)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                subtree=ID3_depth_entropy(depth-1,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree


# In[13]:


def predict(query,tree,default = 1):    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result

def test(data,label,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
        
    return np.sum(predicted["predicted"] == label)/len(data)*100


# In[15]:


print("With the entropy,")
print("--------------------------------------------------")
for i in range(7):
    tree=ID3_depth_entropy(i+1, data, Attributes,'Chance_of_Admit')
    train_acc= test(data,y_train,tree)
    test_acc = test(test_data,y_test,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)


# In[18]:


pprint(tree)


# In[ ]:




