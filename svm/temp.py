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

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
	clf = svm.SVC(kernel='poly', C=1, probability=True).fit(X_train, y_train)
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



f = open('/home/ysj/Downloads/svm/dataset_kim.txt', 'r')

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

		print(float(beat))

		beat_temp.append(float(row[0])-float(beat))

		beat = row[0]

		beat_loudness_temp.append(round(float(row[1]),4))
	
	
	f.close()


