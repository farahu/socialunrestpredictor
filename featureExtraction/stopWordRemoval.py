import sklearn
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import pandas as pd



listOfStopFiles = ["minimal.txt"];
listOfStopWords = [];





def loadStopFiles():
	for stopFile in listOfStopFiles:
		f = open(stopFile, 'w')
		for line in f:
			listOfStopWords.append(line);





print listOfStopWords;
