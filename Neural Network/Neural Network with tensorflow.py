#!/usr/bin/env python
# coding: utf-8

# In[378]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[379]:


# import csv file using panda
Head= ['variance','skewness','curtosis','entropy','label']
data=pd.read_csv('train.csv',names=Head)
test_data=pd.read_csv('test.csv',names= Head)

# replace label "0" to "-1"
#data.label = data.label.replace(0,-1)
#test_data.label = test_data.label.replace(0,-1)


Feature = ['variance','skewness','curtosis','entropy']
X_train = data[Feature]

X_train = np.array(X_train).reshape(-1,4)
X_test = test_data[Feature]

X_test = np.array(X_test).reshape(-1,4)

y = data['label']
y = np.array(y)
y_test = test_data['label']
y_test = np.array(y_test)


# In[380]:



y_train = y
X_valid, X_train = X_train[:620], X_train[620:]
y_valid, y_train = y_train[:620], y_train[620:]


# In[381]:


n_inputs = 4  
n_hidden1 =5
n_hidden2 = n_hidden1
n_outputs = 2


# In[382]:


tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")


# In[383]:


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


# In[384]:


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    
    logits = neuron_layer(hidden2, n_outputs, name="outputs")


# In[385]:


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


# In[386]:


learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# In[387]:


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# In[388]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[389]:


n_epochs = 100
batch_size = 10


# In[390]:


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# In[391]:


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        #acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "train accuracy:", acc_train, "test acc",acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")

