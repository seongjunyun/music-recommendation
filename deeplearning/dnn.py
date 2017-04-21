# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.learn.python.learn as learn
import numpy as np
import pandas as pd

from pandas import DataFrame
import csv


from math import exp

import os
import sys
import filepath

from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics

import shutil

 




def padding(array,max):

	for x in array:

		for i in range(0,max-len(x)):
			x.append(float(0))

def slicing(array,min):

	count = 0
	for x in array:

		x = x[0:100]
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

		beat_temp.append(round(float(row[0])-float(beat),4))
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

		pitch_temp.append(round(float(row[0]),4))
	
	
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

slicing(mfcc,mfccmax)


#padding(pitch,pitchmax)

slicing(pitch,pitchmin)








f = open('/home/ysj/Downloads/svm/target_kim.csv', 'r')
csvReader = csv.reader(f)

targets = []
for row in csvReader:

	targets.append(row)


y = []

for target in targets:

	y.append(int(target[0]))


X_train, X_test, y_train, y_test = train_test_split(beat_loudnesses, y, test_size=0.2, random_state=42)


X_train = np.array(X_train)
y_train = np.array(y_train)

# Specify that all features have real-value data
feature_columns = learn.infer_real_valued_columns_from_input(X_train)
# Build 3 layer DNN with 10, 20, 10 units respectively.

try:
    shutil.rmtree('/home/ysj/Downloads/deeplearning/cronos123_beat_like')
except OSError as e:
    if e.errno == 2:
        # 파일이나 디렉토리가 없음!
        print ('No such file or directory to remove')
        pass
    else:
        raise

classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2, feature_columns=feature_columns,model_dir="/home/ysj/Downloads/deeplearning/cronos123_beat_like")





 

# Fit model.
classifier.fit(x=X_train,
               y=y_train,
               steps=3000)

#y_test2 = y_test
X_test = np.array(X_test)
y_test = np.array(y_test) 
# Evaluate accuracy.
#accuracy_score = classifier.evaluate(x=X_test,
#                                     y=y_test)["accuracy"]
#print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
#new_samples = np.array(
#    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict_proba(X_test, as_iterable=True))

predictions = list(classifier.predict(X_test, as_iterable=True))

result = []
index=0
for prob in y:
	temp = []
	temp.append(prob[1])
	temp.append(y_test[index])
	index = index+1
	result.append(temp)

score = metrics.accuracy_score(y_test, predictions)
print("Accuracy: %f" % score)


print (result)



