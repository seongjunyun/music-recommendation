# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell




import tensorflow as tf
import numpy as np
import pandas as pd

from pandas import DataFrame
import csv


from math import exp

import os
import sys
import filepath


try:
	    file = sys.argv[1]
except:
	    print "usage:", sys.argv[0], "<input-audiofile> <output-audiofile>"
	    sys.exit()


f = open(file, 'r')
csvReader = csv.reader(f)

for row in csvReader:

	beat_temp.append(float(row[0])-float(beat))
	beat = row[0]

	beat_loudness_temp.append(float(row[1]))
	
	
f.close()


# Network Parameters
n_input = 1 # MNIST data input (img shape: 28*28)
n_steps = 800 # timesteps
n_hidden = 60 # hidden layer num of features
n_classes = 2 

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)


print ("bbbb")
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
print ("bbbb")
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print ("bbbb")
# Initializing the variables
init = tf.initialize_all_variables()
print ("bbbb")
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    print ("bbb")
    # Run optimization op (backprop)

    print (X_train[0])
    print (np.array(X_train).shape)

    a = np.array(X_train).reshape(74,n_steps, n_input)

    b = np.array(y_train).reshape(74,n_classes)

    

    sess.run(optimizer, feed_dict={x: a, y: b})


    acc = sess.run(accuracy, feed_dict={x: a, y: b})
    # Calculate batch loss
    loss = sess.run(cost, feed_dict={x: a, y: b})
    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
		  "{:.6f}".format(loss) + ", Training Accuracy= " + \
		  "{:.5f}".format(acc))
	
    print("Optimization Finished!")

