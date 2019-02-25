
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pprint import pprint
import csv


# In[2]:


# Define gain methods
def entropy(target_col):
    elements, counts = np.unique(target_col,return_counts = True)
    entropy = 0
    
    for i in range(len(elements)):    
        p_i = counts[i]/np.sum(counts)
        entropy += -p_i*np.log2(p_i)    
    
    return entropy 

def majority_error(target_col): 
    elements, counts = np.unique(target_col,return_counts= True)
    m_err = 1-counts[np.argmax(counts)]/np.sum(counts)
    
    return m_err

def Gini_index(target_col):
    elements, counts = np.unique(target_col,return_counts= True)
    GI=1
    
    for i in range(len(elements)):
        GI-=(counts[i]/np.sum(counts))**2
    
    return GI
    


# In[3]:


#Compute Information gain and fidn the best attribute
def InfoGain_base(base,data,attribute,label_name="class"):
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
    
    elif base ==2:
        total_m_err = majority_error(data[label_name])
        values, counts = np.unique(data[attribute],return_counts=True)
        
        weighted_m_err = 0 
        for i in range(len(values)):
            sub_data= data[data[attribute]==values[i]]
            weighted_m_err += counts[i]/np.sum(counts)*majority_error(sub_data[label_name])
        
        InfoGain = total_m_err - weighted_m_err

        return InfoGain
    
    elif base == 3:
        total_GI = Gini_index(data[label_name])
        values, counts = np.unique(data[attribute],return_counts=True)
        
        weighted_GI = 0 
        for i in range(len(values)):
            sub_data= data[data[attribute]==values[i]]
            weighted_GI += counts[i]/np.sum(counts)*Gini_index(sub_data[label_name])
            
        InfoGain = total_GI - weighted_GI

        return InfoGain 

# Find the common label which will be the leaf node.  
def Common_label(data, label="class"):
    
    feature, f_count = np.unique(data[label],return_counts= True)
    common_label = feature[np.argmax(f_count)]
    
    return common_label
    


# In[4]:


# ID3_depending on the depth
def ID3_depth_entropy(depth, data,Attributes,label="class"):
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

#ID3 with option to limit depth
def ID3_depth_ME(depth, data,Attributes,label="class"):
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
            item_values=[ InfoGain_base(2,data,f,label) for f in Attributes]

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
                subtree=ID3_depth_ME(depth-1,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree

    #ID3 with option to limit depth
def ID3_depth_GI(depth, data,Attributes,label="class"):
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
            item_values=[ InfoGain_base(3,data,f,label) for f in Attributes]

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
                subtree=ID3_depth_GI(depth-1,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree


# In[5]:


# Apply the ID3 and predict test dataset.

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


# In[6]:


# import "car" data 
# problem2. 
data=pd.read_csv('car_data/train.csv',names= ['buying','maint','doors','persons','lug_boot','safety','labels'])
test_data=pd.read_csv('car_data/test.csv',names= ['buying','maint','doors','persons','lug_boot','safety','labels'])

# Split the data and its labels.
Attributes= ['buying','maint','doors','persons','lug_boot','safety']
Training_Label= data['labels']
Training_Data= data[Attributes]

Test_Label= test_data['labels']
Test_Data= test_data[Attributes]

print("------------------------- Problem2. ----------------------------")
print("With Information Gain,")
print("--------------------------------------------------")
for i in range(6):
    tree=ID3_depth_entropy(i+1, data, Attributes,"labels")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")

#----------------------------------------------------------
print("With the Majority Error,")
print("--------------------------------------------------")
for i in range(6):
    tree=ID3_depth_ME(i+1, data, Attributes,"labels")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")
#-------------------------------------------------------------
print("With the Gini Index,")
print("--------------------------------------------------")
for i in range(6):
    tree=ID3_depth_GI(i+1, data, Attributes,"labels")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")
    


# In[7]:


# import csv file using panda
Head= ['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome','Label']
data=pd.read_csv('bank_data/train.csv',names=Head)
test_data=pd.read_csv('bank_data/test.csv',names= Head)

print(data.head())


# In[8]:


def conv_num_to_bi(data,test_data, target_head="class"):
    med_d = data[target_head].median()
    min_d = data[target_head].min()
    max_d = data[target_head].max()
    bins = [min_d-1,med_d,max_d]
    data[target_head]=pd.cut(data[target_head],bins,labels=[0,1]) 
    test_data[target_head]=pd.cut(test_data[target_head],bins,labels=[0,1])


# In[9]:



# convert numeric data to binary
# train
conv_num_to_bi(data,test_data,"Age")
conv_num_to_bi(data,test_data,"Balance")
conv_num_to_bi(data,test_data,"Day")
conv_num_to_bi(data,test_data,"Duration")
conv_num_to_bi(data,test_data,"Campaign")
conv_num_to_bi(data,test_data,"Pdays")
conv_num_to_bi(data,test_data,"Previous")


# In[10]:



# Split the data and its labels.
Attributes= ['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome']
Training_Label= data['Label']
Training_Data= data[Attributes]

Test_Label= test_data['Label']
Test_Data= test_data[Attributes]


# In[11]:


print("-------------------------3.(a)----------------------------")
print("With Information Gain,")
print("--------------------------------------------------")
for i in range(16):
    tree=ID3_depth_entropy(i+1, data, Attributes,"Label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")

#----------------------------------------------------------
print("With the Majority Error,")
print("--------------------------------------------------")
for i in range(16):
    tree=ID3_depth_ME(i+1, data, Attributes,"Label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")
#-------------------------------------------------------------
print("With the Gini Index,")
print("--------------------------------------------------")
for i in range(16):
    tree=ID3_depth_GI(i+1, data, Attributes,"Label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")


# In[12]:



#--------------with unknown----------------------------------
def Common_value(data, label="class"):
    """ data : given dataset
        label: title of the label"""
    feature, f_count = np.unique(data[label],return_counts= True)
    max_count= np.argmax(f_count)
    if feature[max_count]=="unknown":
        f_count[max_count]=0
        common_label= feature[np.argmax(f_count)]
    else: 
        common_label = feature[max_count]
        
    
    return common_label
# Convert "unknown" values to the common values in the attributes.
com_Job= Common_value(data, label="Job")
com_Education= Common_value(data, label="Education")
com_Poutcome= Common_value(data, label="Poutcome")
com_Contact= Common_value(data, label="Contact")

data["Poutcome"]=data.Poutcome.replace("unknown", com_Poutcome)
data["Job"]=data.Job.replace("unknown", com_Job)
data["Education"]=data.Education.replace("unknown", com_Education)
data["Contact"]=data.Contact.replace("unknown", com_Contact)

test_data["Poutcome"]=test_data.Poutcome.replace("unknown", com_Poutcome)
test_data["Job"]=test_data.Job.replace("unknown", com_Job)
test_data["Education"]=test_data.Education.replace("unknown", com_Education)
test_data["Contact"]=test_data.Contact.replace("unknown", com_Contact)

# Split the data and its labels.
Attributes= ['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome']
Training_Label= data['Label']
Training_Data= data[Attributes]

Test_Label= test_data['Label']
Test_Data= test_data[Attributes]
#-------------------------------------------------------------------------------------------------------------
print("-------------------------3.(b)----------------------------")
print("With the Information Gain,")
print("--------------------------------------------------")
for i in range(16):
    tree=ID3_depth_entropy(i+1, data, Attributes,"Label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")

#----------------------------------------------------------
print("With the Majority Error,")
print("--------------------------------------------------")
for i in range(16):
    tree=ID3_depth_ME(i+1, data, Attributes,"Label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")
#-------------------------------------------------------------
print("With the Gini Index,")
print("--------------------------------------------------")
for i in range(16):
    tree=ID3_depth_GI(i+1, data, Attributes,"Label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    

    print("depth is ", i+1)
    print ("training acc is ", train_acc)
    print("testing acc is ", test_acc)
    print("----------------------------------------------")

