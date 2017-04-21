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


clf = joblib.load('beat.pkl') 

print (clf)





