import sklearn
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import pandas as pd
import random
import time
from datetime import datetime
from time import mktime
import sys

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/GetOldTweets-python");
from Exporter import tweetQuery




listOfStopFiles = ["minimal.txt"];
listOfStopWords = [];
listOfRandomDates = [];




def rangeTimeProp(start, end, format, prop):

	stime = time.mktime(time.strptime(start, format))
	etime = time.mktime(time.strptime(end, format))

	ptime = stime + prop * (etime - stime)

	dt = datetime.fromtimestamp(ptime);

	#return (dt, dt + timedelta(days=1));

	return (time.strftime(format, time.localtime(ptime)), time.strftime(format, time.localtime(ptime + 100000)));


def randomDate(start, end, prop):

	return (rangeTimeProp(start, end, '%Y-%m-%d', prop))

#print randomDate("1/1/2016", "09/1/2016", random.random())



def generateRandomTweets():

	for x in range(0, 10):
		dateRange = randomDate("2016-01-01", "2016-01-09", random.random());
		argv = ["--since", dateRange[0], "--until", dateRange[1], "--maxtweets", '1000'];
		tweetQuery(argv, "tweetsForStopWord" + dateRange[0] + ".csv");





		#date2 = date[date.find("/");












generateRandomTweets();
def loadStopFiles():
	for stopFile in listOfStopFiles:
		print stopFile
		f = open(stopFile, 'r')
		print f
		for line in f:
			line = line[:-1];
			listOfStopWords.append(line);



#def generateStopWords():









	#Generate random dates
	#Run the Python Tweet Script
	#Load CSV the CSV File
	#Store the tweets in string format











loadStopFiles();

print listOfStopWords;
