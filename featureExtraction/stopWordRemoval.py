import collections
import sklearn
import operator
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import pandas as pd
import random
import time
import csv
from datetime import datetime
from time import mktime
import sys
import os

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/GetOldTweets-python");
from Exporter import tweetQuery




listOfStopFiles = ["finalStopWords.txt"];
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
                print "Testing"
                print x;




		#date2 = date[date.find("/");












# generateRandomTweets();
def loadStopFiles():
	for stopFile in listOfStopFiles:
		print stopFile
		f = open(stopFile, 'r')
		print f
		for line in f:
			line = line[:-1];
			listOfStopWords.append(line);



def loadTweetFiles():
    tweetArray = []
    for filename in os.listdir('../TweetsForStopWords'):
        with open('../TweetsForStopWords/' + filename, "rb") as csvfile:
            tweets = csv.reader(csvfile, delimiter=";", quotechar="|")
            for row in tweets:
		tweetArray.append(row[4].lower())
    return tweetArray


def generateStopWords():
	loadStopFiles()
	print 'stop word list so far ', listOfStopWords

	tweetArray = []
	wordCount = collections.defaultdict(int)
	for tweet in tweetArray:
		words = tweet.split()
		for word in words:
			if word in listOfStopWords:
				continue
			wordCount[word] += 1

	sortedWordCount = sorted(wordCount.items(), key=operator.itemgetter(1), reverse=True)

        for index in range(38):
		listOfStopWords.append(sortedWordCount[index][0])

	text_file = open("finalStopWords.txt", "w")
	for word in listOfStopWords:
		text_file.write(word + "\n")
	text_file.close()

	print listOfStopWords





	#Generate random dates
	#Run the Python Tweet Script
	#Load CSV the CSV File
	#Store the tweets in string format









# loadStopFiles();
# generateStopWords();


modifiedTweets = [];

def removeStopWords():
    loadStopFiles()
    tweetArray = loadTweetFiles()
    modifiedTweetArray = []
    for tweet in tweetArray:
        modifiedTweet = tweet.split()
        for word in listOfStopWords:
            if word in modifiedTweet:
                modifiedTweet.remove(word)

        modifiedTweetArray.append(modifiedTweet)
    text_file = open("stoppedTweets.txt", "w")
    for tweet in modifiedTweetArray:
	text_file.write(' '.join(tweet) + "\n")
    text_file.close()
    print modifiedTweetArray
removeStopWords()
#print tweetArray;


#print listOfStopWords;
