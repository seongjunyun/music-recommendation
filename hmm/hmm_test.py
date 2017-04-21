# -*- coding: utf-8 -*-
from __future__ import print_function

import datetime

import numpy as np

from hmmlearn.hmm import GaussianHMM

import csv

from yahmm import*

from math import exp

import os
import sys
import filepath


folder = os.getcwd();
files = filepath.allfiles(folder)

i=0	

beats = []

beat_loudnesses = []

for file in files:

	temp = file.split('.')
	length = len(temp)
	temp2 = temp[length-2].split('/')
	length2 = len(temp2)

	

	print (file)
	
	if(temp[length-1] == "csv" ):

		
	
		f = open(file, 'r')
		csvReader = csv.reader(f)

		beat_temp = []
		beat_loudness_temp = []

		beat = 0.0

		for row in csvReader:
			
			beat_temp.append(float(row[0])-float(beat))
			beat = row[0]

			beat_loudness_temp.append(float(row[1]))
			
		f.close()
		
		beats.append(beat_temp)
		beat_loudnesses.append(beat_loudness_temp)
		
matrix = []
matrix2 = []
lengths = []
lengths2 = []

a = 0
for beat in beats:
	lengths.append(len(beat))
	matrix = matrix+beat


for beat_loudness in beat_loudnesses:
	lengths2.append(len(beat_loudness))
	matrix2 = matrix2+beat_loudness

print("fitting to HMM and decoding ...", end="")

model = GaussianHMM(n_components=1, covariance_type="spherical", n_iter=1000).fit(np.atleast_2d(matrix).T,lengths)




f = open('/home/ysj/Downloads/어쿠스틱 콜라보-그대와 나, 설레임 (Feat. 소울맨)_percussive_beat.csv', 'r')
csvReader = csv.reader(f)

test = []

beat = 0.0
for row in csvReader:
			
	test.append(float(row[0])-float(beat))
	beat = row[0]
	
f.close()

f = open('/home/ysj/Downloads/샵건-미친놈 (Feat. 제시)_percussive_beat.csv', 'r')
csvReader = csv.reader(f)

test2 = []

beat = 0.0
for row in csvReader:
			
	test2.append(float(row[0])-float(beat))
	beat = row[0]
	
f.close()

f = open('/home/ysj/Downloads/David Guetta-Bad (Feat. Vassy) (Radio Edit)_percussive_beat.csv', 'r')
csvReader = csv.reader(f)

test3 = []

beat = 0.0
for row in csvReader:
			
	test3.append(float(row[0])-float(beat))
	beat = row[0]
	
f.close()

f = open('/home/ysj/Downloads/Adele-Hello_percussive_beat.csv', 'r')
csvReader = csv.reader(f)

test4 = []

beat = 0.0
for row in csvReader:
			
	test4.append(float(row[0])-float(beat))
	beat = row[0]
	
f.close()


f = open('/home/ysj/Downloads/여자친구-02-시간을 달려서 (Rough)_percussive_beat.csv', 'r')
csvReader = csv.reader(f)

test5 = []

beat = 0.0
for row in csvReader:
			
	test5.append(float(row[0])-float(beat))
	beat = row[0]
	
f.close()






logprob = model.score(beats[0])

logprob2 = model.score(test)

logprob3 = model.score(test2)

logprob4 = model.score(test3)

logprob5 = model.score(test4)

logprob6 = model.score(test5)




print("")

print ("어쿠스틱 콜라보-그대와 나, 설레임 (Feat. 소울맨)",logprob2)

print ("샵건 - 미친놈",logprob3)

print ("박효신- 숨",logprob)

print ("David Guetta-Bad (Feat. Vassy) (Radio Edit)",logprob4)


print ("Adele-Hello",logprob5)

print ("여자친구-02-시간을 달려서",logprob6)







