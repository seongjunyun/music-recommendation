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

from sklearn.model_selection import train_test_split


def padding(array,max):

	for x in array:

		for i in range(0,max-len(x)):
			x.append(float(0))

def slicing(array,min):

	count = 0
	for x in array:

		x = x[0:min]
		array[count] = x
		count = count+1

		

folder = os.getcwd();
files = filepath.allfiles(folder)

i=0	

beats = []

beat_loudnesses = []

mfcc = []

pitch = []




f = open('/home/ysj/Downloads/svm/dataset_kim.txt', 'r')

files = []

while True:
    line = f.readline()
    if not line: break

    line = line.replace("\n","")
    files.append(line)
f.close()


beat = 0.0




beatmax = 0

mfccmax = 0

pitchmax = 0

pitchmin = 10000000

for name in files:
	
	print (name)

	beat_temp = []
	beat_loudness_temp = []

	mfcc_temp = []


	pitch_temp = []

	filename = '/home/ysj/Downloads/dataset/'+name+'_percussive_beat.csv'
	f = open(filename, 'r')
	csvReader = csv.reader(f)

	for row in csvReader:

		beat_temp.append(float(row[0])-float(beat))
		beat = row[0]

		beat_loudness_temp.append(round(float(row[1]),4))
	
	
	f.close()

	filename = '/home/ysj/Downloads/dataset/'+name+'_mfcc.csv'
	f = open(filename, 'r')
	csvReader = csv.reader(f)

	for row in csvReader:

		temp = []
		temp.append(round(float(row[0]),4))
		temp.append(round(float(row[1]),4))
		temp.append(round(float(row[2]),4))
		temp.append(round(float(row[3]),4))
		temp.append(round(float(row[4]),4))
		temp.append(round(float(row[5]),4))
		temp.append(round(float(row[6]),4))
		temp.append(round(float(row[7]),4))
		temp.append(round(float(row[8]),4))
		temp.append(round(float(row[9]),4))
		temp.append(round(float(row[10]),4))
		temp.append(round(float(row[11]),4))
		temp.append(round(float(row[12]),4))

		mfcc_temp.append(temp)
	
	
	f.close()

	filename = '/home/ysj/Downloads/dataset/'+name+'_pitch.csv'
	f = open(filename, 'r')
	csvReader = csv.reader(f)

	for row in csvReader:

		pitch_temp.append(float(row[0]))
	
	
	f.close()


	

	if(len(beat_temp) > beatmax):
		beatmax = len(beat_temp)

	if(len(mfcc_temp) > mfccmax):
		mfccmax = len(mfcc_temp)

	if(len(pitch_temp) > pitchmax):
		pitchmax = len(pitch_temp)

	if(len(pitch_temp) < pitchmin):
		pitchmin = len(pitch_temp)



	beats.append(beat_temp)
	beat_loudnesses.append(beat_loudness_temp)
	mfcc.append(mfcc_temp)

	pitch.append(pitch_temp)
	
f.close()

padding(beats,beatmax)

padding(beat_loudnesses,beatmax)

padding(mfcc,mfccmax)


#padding(pitch,pitchmax)


slicing(beats,800)
slicing(beat_loudnesses,800)
slicing(mfcc,800)
slicing(pitch,800)








f = open('/home/ysj/Downloads/svm/target_kim.csv', 'r')
csvReader = csv.reader(f)

targets = []
for row in csvReader:

	targets.append(row)


y = []

for target in targets:

	if int(target[0])==0:
		temp = [1,0]
		y.append(temp)
	else:
		temp = [0,1]
		y.append(temp)


print ("aaaaa")

X_train, X_test, y_train, y_test = train_test_split(beats, y, test_size=0.2, random_state=42)

print("bbbb")

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

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

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]

    a = np.array(X_test).reshape(19,n_steps, n_input)

    b = np.array(y_test).reshape(19,n_classes)
    print("Testing Accuracy:", \

    sess.run(accuracy, feed_dict={x: a, y: b}))

    print(sess.run(pred, feed_dict={x: a}))
    
    

    
