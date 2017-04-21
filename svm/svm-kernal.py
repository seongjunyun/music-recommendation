# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.externals import joblib




import datetime

from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
try:
    from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
    # For Matplotlib prior to 1.5.
    from matplotlib.finance import (
        quotes_historical_yahoo as quotes_historical_yahoo_ochl
    )

from hmmlearn.hmm import GaussianHMM

import csv

from yahmm import*

from math import exp

import os
import sys
import filepath
from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy







def padding(array,max):

	for x in array:

		for i in range(0,max-len(x)):
			x.append(float(0))

def slicing(array,min):

	count = 0
	for x in array:

		x = x[100:130]
		array[count] = x
		count = count+1

		



def train_test(x,y):

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	clf = svm.SVC(kernel='rbf', C=1, probability=True).fit(X_train, y_train)
	#print("result", clf.predict(X_test))
	#print("result_log", clf.predict_log_proba(X_test))

	score = clf.score(X_test, y_test) 

	print (score)   

	print (numpy.array(X_test))                    

	score2 = clf.predict_proba(X_test)

	print (score2)

	print (y_test)

	return clf



folder = os.getcwd();
files = filepath.allfiles(folder)

i=0	

beats = []

beat_loudnesses = []

mfcc1 = []
mfcc2 = []
mfcc3 = []
mfcc4 = []
mfcc5 = []
mfcc6 = []
mfcc7 = []
mfcc8 = []
mfcc9 = []
mfcc10 = []
mfcc11 = []
mfcc12 = []
mfcc13 = []

pitch = []

tempogram = []


f = open('/home/ysj/Downloads/svm/dataset_temp.txt', 'r')

files = []

while True:
    line = f.readline()
    if not line: break

    line = line.replace("\n","")
    files.append(line)
f.close()





beatmax = 0

mfccmax = 0

pitchmax = 0

pitchmin = 10000000

for name in files:
	
	print (name)

	beat_temp = []
	beat_loudness_temp = []

	mfcc1_temp = []
	mfcc2_temp = []
	mfcc3_temp = []
	mfcc4_temp = []
	mfcc5_temp = []
	mfcc6_temp = []
	mfcc7_temp = []
	mfcc8_temp = []
	mfcc9_temp = []
	mfcc10_temp = []
	mfcc11_temp = []
	mfcc12_temp = []
	mfcc13_temp = []

	pitch_temp = []

	filename = '/home/ysj/Downloads/dataset/'+name+'_percussive_beat.csv'
	f = open(filename, 'r')
	csvReader = csv.reader(f)

	beat = 0.0

	for row in csvReader:

		beat_temp.append(round(float(row[0])-float(beat),4))
		beat = row[0]

		beat_loudness_temp.append(round(float(row[1]),4))
	
	
	f.close()

	filename = '/home/ysj/Downloads/dataset/'+name+'_mfcc.csv'
	f = open(filename, 'r')
	csvReader = csv.reader(f)

	for row in csvReader:

		mfcc1_temp.append(round(float(row[0]),4))
		mfcc2_temp.append(round(float(row[1]),4))
		mfcc3_temp.append(round(float(row[2]),4))
		mfcc4_temp.append(round(float(row[3]),4))
		mfcc5_temp.append(round(float(row[4]),4))
		mfcc6_temp.append(round(float(row[5]),4))
		mfcc7_temp.append(round(float(row[6]),4))
		mfcc8_temp.append(round(float(row[7]),4))
		mfcc9_temp.append(round(float(row[8]),4))
		mfcc10_temp.append(round(float(row[9]),4))
		mfcc11_temp.append(round(float(row[10]),4))
		mfcc12_temp.append(round(float(row[11]),4))
		mfcc13_temp.append(round(float(row[12]),4))
	
	
	f.close()

	filename = '/home/ysj/Downloads/dataset/'+name+'_pitch.csv'
	f = open(filename, 'r')
	csvReader = csv.reader(f)

	for row in csvReader:

		pitch_temp.append(round(float(row[0]),4))
	
	
	f.close()


	

	if(len(beat_temp) > beatmax):
		beatmax = len(beat_temp)

	if(len(mfcc1_temp) > mfccmax):
		mfccmax = len(mfcc1_temp)

	if(len(pitch_temp) > pitchmax):
		pitchmax = len(pitch_temp)

	if(len(pitch_temp) < pitchmin):
		pitchmin = len(pitch_temp)



	beats.append(beat_temp)
	beat_loudnesses.append(beat_loudness_temp)
	mfcc1.append(mfcc1_temp)
	mfcc2.append(mfcc2_temp)
	mfcc3.append(mfcc3_temp)
	mfcc4.append(mfcc4_temp)
	mfcc5.append(mfcc5_temp)
	mfcc6.append(mfcc6_temp)
	mfcc7.append(mfcc7_temp)
	mfcc8.append(mfcc8_temp)
	mfcc9.append(mfcc9_temp)
	mfcc10.append(mfcc10_temp)
	mfcc11.append(mfcc11_temp)
	mfcc12.append(mfcc12_temp)
	mfcc13.append(mfcc13_temp)

	pitch.append(pitch_temp)
	
f.close()

padding(beats,beatmax)

padding(beat_loudnesses,beatmax)

slicing(mfcc1,mfccmax)
padding(mfcc2,mfccmax)
padding(mfcc3,mfccmax)
padding(mfcc4,mfccmax)
padding(mfcc5,mfccmax)
padding(mfcc6,mfccmax)
padding(mfcc7,mfccmax)
padding(mfcc8,mfccmax)
padding(mfcc9,mfccmax)
padding(mfcc10,mfccmax)
padding(mfcc11,mfccmax)
padding(mfcc12,mfccmax)
padding(mfcc13,mfccmax)

#padding(pitch,pitchmax)

slicing(pitch,pitchmin)








f = open('/home/ysj/Downloads/svm/target_temp.csv', 'r')
csvReader = csv.reader(f)

targets = []
for row in csvReader:

	targets.append(row)


y = []

for target in targets:

	y.append(target[0])


print("beat:")

clf = train_test(beats,y)
joblib.dump(clf, 'beat.pkl') 

print("")
print("beat_loudness:")
clf = train_test(beat_loudnesses,y)
joblib.dump(clf, 'beat_loudness.pkl') 

print("")
print("mfcc1:")
clf = train_test(mfcc1,y)
joblib.dump(clf, 'mfcc1.pkl') 

print("")
print("mfcc2:")
clf = train_test(mfcc2,y)
joblib.dump(clf, 'mfcc2.pkl') 

print("")
print("mfcc3:")
clf = train_test(mfcc3,y)
joblib.dump(clf, 'mfcc3.pkl') 

print("")
print("mfcc4:")
clf = train_test(mfcc4,y)
joblib.dump(clf, 'mfcc4.pkl')

print("")
print("mfcc5:")
clf = train_test(mfcc5,y)
joblib.dump(clf, 'mfcc5.pkl')

print("")
print("mfcc6:")
clf = train_test(mfcc6,y)
joblib.dump(clf, 'mfcc6.pkl')

print("")
print("mfcc7:")
clf = train_test(mfcc7,y)
joblib.dump(clf, 'mfcc7.pkl')

print("")
print("mfcc8:")
clf = train_test(mfcc8,y)
joblib.dump(clf, 'mfcc8.pkl')

print("")
print("mfcc9:")
clf = train_test(mfcc9,y)
joblib.dump(clf, 'mfcc9.pkl')

print("")
print("mfcc10:")
clf = train_test(mfcc10,y)
joblib.dump(clf, 'mfcc10.pkl')

print("")
print("mfcc11:")
clf = train_test(mfcc11,y)
joblib.dump(clf, 'mfcc11.pkl')

print("")
print("mfcc12:")
clf = train_test(mfcc12,y)
joblib.dump(clf, 'mfcc12.pkl')

print("")
print("mfcc13:")
clf = train_test(mfcc13,y)
joblib.dump(clf, 'mfcc13.pkl')



print("")
print("pitch:")
clf = train_test(pitch,y)
joblib.dump(clf, 'pitch.pkl')






