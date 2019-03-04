
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
from pprint import pprint
import csv
import matplotlib.pyplot as plt


# In[37]:


def conv_num_to_bi(data,test_data, target_head="class"):
    med_d = data[target_head].median()
    min_d = data[target_head].min()
    max_d = data[target_head].max()
    bins = [min_d-1,med_d,max_d]
    data[target_head]=pd.cut(data[target_head],bins,labels=[0,1]) 
    test_data[target_head]=pd.cut(test_data[target_head],bins,labels=[0,1])


# In[38]:


Head= ["Age","Job","Marital","Education","Default","Balance","Housing","Loan","Contact","Day","Month","Duration","Campaign","Pday","Previous","Poutcome","Label"]
Data=pd.read_csv('train.csv',names=Head)
test_Data=pd.read_csv('test.csv',names= Head)
       

Attributes=["Age","Job","Marital","Education","Default","Balance","Housing","Loan","Contact","Day","Month","Duration","Campaign","Pday","Previous","Poutcome"]
train_Data= Data[Attributes]
train_Label = Data["Label"]


n_data, dim = train_Data.shape
data = pd.DataFrame(Data.copy())
test_data =  pd.DataFrame(test_Data.copy())
# convert numeric data to binary
# train
conv_num_to_bi(data,test_data,"Age")
conv_num_to_bi(data,test_data,"Balance")
conv_num_to_bi(data,test_data,"Day")
conv_num_to_bi(data,test_data,"Duration")
conv_num_to_bi(data,test_data,"Campaign")
conv_num_to_bi(data,test_data,"Pday")
conv_num_to_bi(data,test_data,"Previous")


# In[40]:


# Assign the weights
data["weights"] = 1/len(data)
test_data["weights"] = 1/len(data)
#names of label should be "label"
data["label"] = data.Label
data["label"]=data.label.replace("no",-1)
data["label"]=data.label.replace("yes",1)
test_data["label"] = test_data.Label
test_data["label"]=test_data.label.replace("no",-1)
test_data["label"]=test_data.label.replace("yes",1)


# In[41]:


def weighted_entropy(data,target): 
    p_label, p_count = np.unique(data[target],return_counts =True)
    prop=[]

    for f in p_label:
        p=0
        for i in data.index:
            if data[target][i]==f:
                p += data["weights"][i]
            else:
                p+= 0
                
        prop.append(p)
    
    entro=0
    for i in range(len(prop)):
        p= prop[i]/np.sum(prop)
        entro += -p*np.log(p)
        
    return entro

# weighted_info_gain(data,"attribute")

def weigthed_info_gain(data, attribute):
    total_entro = weighted_entropy(data,"label")
    

    feature, f_counts = np.unique(data[attribute],return_counts = True)

    weighted_entro = 0 

    for f in feature:
        sub_data= data[data[attribute]==f]
        
        p = weighted_entropy(sub_data,"label")
        weighted_entro +=p*len(sub_data)/len(data)
        

    info_gain = total_entro - weighted_entro
    
    return info_gain

def Common_label(data):
    
    feature,f_count = np.unique(data["label"],return_counts =True)
    W_counts = []
    for f in feature:
        sdata=data[data["label"]==f]
        weighted_count = np.sum(sdata["weights"])
        W_counts.append(weighted_count)

    common_label = feature[np.argmax(W_counts)]
    
    return common_label


# In[42]:


def ID3_depth_entropy(depth, data,Attributes,label="class"):
    
    common_label=Common_label(data)
    
    # if all label_values have same value, return itself
    if len(np.unique(data[label])) <= 1:
        return common_label
        #return 

    # if feature space is empty, return the common label value
    elif len(Attributes)==0:
        return common_label

    # go to the scheme
    else:
       # select the attribute which best split the dataset using InfoGain
        
        item_values=[ weigthed_info_gain(data, f) for f in Attributes]

        best_attribute_index = np.argmax(item_values)
        best_attribute = Attributes[best_attribute_index]
        
        tree = {best_attribute:{}}

        
        # grow a branch under the root node
        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute]== value]
            sub_common_label= Common_label(sub_data)

            if len(sub_data)==0 or depth==1:
                tree[best_attribute][value] = sub_common_label

            else:
                Attributes = [ i for i in Attributes if i != best_attribute]
                subtree=ID3_depth_entropy(depth-1,sub_data,Attributes,"label")
                tree[best_attribute][value]=subtree
        return tree

def predict(query,tree,default = 1):    
    
    for key in list(query.keys()):
        if key in list(tree.keys()):
          
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
        


def test(data,tree):
    label = data["label"]
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
        
    return predicted, np.sum(predicted["predicted"] != label)/len(data)*100

    


# In[43]:


#Q2.(a) Boosting
data["weights"] = 1/len(data)
Training_Error =[]
each_Error =[]
Testing_Error = []
test_each_Error=[]
n_iter = 0 
Final_pred=np.zeros((len(data),))
Test_Final_pred= np.zeros((len(test_data),))
for i in range(1000):
    epsilon = 0 
    alpha = 0
    tree = ID3_depth_entropy(1, data, Attributes,"Label")

    data["prediction"],acc =  test(data,tree)
    each_Error.append(acc)
    test_data["prediction"],test_acc =  test(test_data,tree)
    test_each_Error.append(test_acc)

    for i in range(len(data)):
        if data["prediction"][i]!= data["label"][i]:
            epsilon += data["weights"][i]

    alpha = np.log((1-epsilon)/epsilon)/2

    data["weights"] *= np.exp(-alpha*data["label"]*data["prediction"])
    data["weights"] = data["weights"]/np.sum(data["weights"])
    Final_pred += alpha*data["prediction"]
    Test_Final_pred += alpha*test_data["prediction"]

    n_iter +=1
    error = np.sum(np.sign(Final_pred) != data["label"])/len(data)*100
    
    test_error = np.sum(np.sign(Test_Final_pred) != test_data["label"])/len(test_data)*100
    Training_Error.append(error)
    Testing_Error.append(test_error)
    print(n_iter,error,test_error)
    print(n_iter,acc,test_acc)
    print("---------------------------------------------------------")


# In[22]:


# draw figure 
plt.figure()
print('---------Boosting Error for H_final and for h_t ---------------------')
plt.plot(range(n_iter),Training_Error,'b.')
plt.plot(range(n_iter),Testing_Error,'g.')

plt.figure()

plt.plot(range(n_iter),each_Error,'b.')
plt.plot(range(n_iter),test_each_Error,'g.')


# In[23]:


def entropy(target_col):
    elements, counts = np.unique(target_col,return_counts = True)
    entropy = 0
    
    for i in range(len(elements)):    
        p_i = counts[i]/np.sum(counts)
        entropy += -p_i*np.log2(p_i)    
    
    return entropy 

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

def Common_label(data, label="class"):
    
    feature, f_count = np.unique(data[label],return_counts= True)
    common_label = feature[np.argmax(f_count)]
    
    return common_label


# In[24]:


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
                Attributes = [i for i in Attributes if i != best_attribute]
                subtree=ID3_depth_entropy(depth-1,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree


# In[25]:


# import csv file using panda
Head= ['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome','Label']
data=pd.read_csv('train.csv',names=Head)
test_data=pd.read_csv('test.csv',names= Head)

data["Label"]=data.Label.replace("no",-1)
data["Label"]=data.Label.replace("yes",1)

test_data["Label"]=test_data.Label.replace("no",-1)
test_data["Label"]=test_data.Label.replace("yes",1)

# convert numeric data to binary
# train
conv_num_to_bi(data,test_data,"Age")
conv_num_to_bi(data,test_data,"Balance")
conv_num_to_bi(data,test_data,"Day")
conv_num_to_bi(data,test_data,"Duration")
conv_num_to_bi(data,test_data,"Campaign")
conv_num_to_bi(data,test_data,"Pdays")
conv_num_to_bi(data,test_data,"Previous")

# Split the data and its labels.
Attributes= ['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome']
Training_Label= data['Label']
Training_Data= data[Attributes]

Test_Label= test_data['Label']
Test_Data= test_data[Attributes]


# In[27]:


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
        
    return predicted["predicted"] ,np.sum(predicted["predicted"] == label)/len(data)*100


# In[28]:


#Q2.(b) begging
sample_num=len(data)

Training_error=[]
Test_error=[]
n_iter=0
total_pred_h=0
test_total_pred_h =0
for i in range(1000):
    sub_index=np.random.choice(len(data), sample_num)
    sub_data=data.iloc[sub_index]
    sub_data.index=[i for i in range(sample_num)]
    
    tree=ID3_depth_entropy(16, sub_data, Attributes,"Label")
    n_iter += 1
    
    pred_h,train_acc= test(data,Training_Label,tree)
    test_pred_h, test_acc = test(test_data,Test_Label,tree)
    
    total_pred_h += pred_h
    avg_pred_h = np.sign(total_pred_h/n_iter)
    
    test_total_pred_h+= test_pred_h
    test_avg_pred_h = np.sign(test_total_pred_h/n_iter)
    
    training_error = np.sum(np.sign(avg_pred_h) != data["Label"])/len(data)*100
    Training_error.append(training_error)
    
    testing_error = np.sum(np.sign(test_avg_pred_h) != test_data["Label"])/len(test_data)*100
    Test_error.append(testing_error)

    print(n_iter,training_error,testing_error)
  
    print("----------------------------------------------------------")


    


# In[29]:


plt.figure()
plt.plot(range(n_iter),Training_error,'b-')
plt.plot(range(n_iter),Test_error,'g-')


# In[30]:


# Define gain methods
def entropy(target_col):
    elements, counts = np.unique(target_col,return_counts = True)
    entropy = 0
    
    for i in range(len(elements)):    
        p_i = counts[i]/np.sum(counts)
        entropy += -p_i*np.log2(p_i)    
    
    return entropy 

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
    
    
# Find the common label which will be the leaf node.  
def Common_label(data, label="class"):
    
    feature, f_count = np.unique(data[label],return_counts= True)
    common_label = feature[np.argmax(f_count)]
    
    return common_label


# In[31]:


# ID3_depending on the depth
def rand_ID3_depth_entropy(depth,sample_num, data, Attributes,label="class"):
    common_label=Common_label(data,label)
    

    # if all label_values have same value, return itself
    if len(np.unique(data[label])) <= 1:
        return common_label
        #return 

    # if feature space is empty, return the common label value
    elif len(Attributes)==0:
        return common_label

    # go to the scheme
    else:
       # select the attribute which best split the dataset using InfoGain
        sub_index=np.random.choice(len(Attributes), sample_num)

        sub_att=[]

        for i in range(len(sub_index)):
            a=Attributes[sub_index[i]]
            sub_att.append(a)



        item_values=[ InfoGain_base(1,data,f,label) for f in sub_att]

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
                Attributes = [i for i in Attributes if i != best_attribute]
                subtree=rand_ID3_depth_entropy(depth-1,sample_num,sub_data,Attributes,label)
                tree[best_attribute][value]=subtree
        return tree

#


# In[33]:


# import csv file using panda
Head= ['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome','Label']
data=pd.read_csv('train.csv',names=Head)
test_data=pd.read_csv('test.csv',names= Head)

data["Label"]=data.Label.replace("no",-1)
data["Label"]=data.Label.replace("yes",1)

test_data["Label"]=test_data.Label.replace("no",-1)
test_data["Label"]=test_data.Label.replace("yes",1)


# convert numeric data to binary
# train
conv_num_to_bi(data,test_data,"Age")
conv_num_to_bi(data,test_data,"Balance")
conv_num_to_bi(data,test_data,"Day")
conv_num_to_bi(data,test_data,"Duration")
conv_num_to_bi(data,test_data,"Campaign")
conv_num_to_bi(data,test_data,"Pdays")
conv_num_to_bi(data,test_data,"Previous")

# Split the data and its labels.
Attributes= ['Age','Job','Marital','Education','Default','Balance','Housing','Loan','Contact','Day','Month','Duration','Campaign','Pdays','Previous','Poutcome']
Training_Label= data['Label']
Training_Data= data[Attributes]

Test_Label= test_data['Label']
Test_Data= test_data[Attributes]


# In[35]:


#random forest
for n_att in [2,4,6]:
    print("-------when n_att is ", n_att)
    sample_num=len(data)
    Training_error=[]
    Test_error=[]
    n_iter=0
    total_pred_h=0
    test_total_pred_h =0
    for i in range(1000):
        sub_index=np.random.choice(len(data), sample_num)
        sub_data=data.iloc[sub_index]
        sub_data.index=[i for i in range(sample_num)]

        tree=rand_ID3_depth_entropy(n_att, n_att, sub_data, Attributes,"Label")
        n_iter += 1

        pred_h,train_acc= test(data,Training_Label,tree)
        test_pred_h, test_acc = test(test_data,Test_Label,tree)

        total_pred_h += pred_h
        avg_pred_h = total_pred_h/n_iter

        test_total_pred_h+= test_pred_h
        test_avg_pred_h = test_total_pred_h/n_iter

        training_error = np.sum(np.sign(avg_pred_h) != data["Label"])/len(data)*100
        Training_error.append(training_error)

        testing_error = np.sum(np.sign(test_avg_pred_h) != test_data["Label"])/len(test_data)*100
        Test_error.append(testing_error)

        print(n_iter,training_error,testing_error)
        print("----------------------------------------------------------")
    print("--------------figure when n_att is ",n_att,"-----------------------")
    figure()
    plt.plot(range(n_iter),Training_error,'b-')
    plt.plot(range(n_iter),Test_error,'g-')
    print("--------------------------------------------------------------------")

