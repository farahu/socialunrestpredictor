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
from UserString import MutableString

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/GetOldTweetsModule")
# from Exporter import tweetQuery


listOfStopFileNames = ["finalStopWords.txt"]
listOfStopWords = []
listOfRandomDates = []
stopWordGenerationDir = "TweetsForStopWords"
punctuation = set([",", ":", "?", ".", "-", "(", ")", "\'", "\"", "!", "/", "#"])

def removePunctuation(word):
    newWord = ""
    for character in word:
        if character in punctuation:
            continue
        newWord += character
    return newWord

def rangeTimeProp(start, end, format, prop):

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    dt = datetime.fromtimestamp(ptime)

    #return (dt, dt + timedelta(days=1))

    return (time.strftime(format, time.localtime(ptime)), time.strftime(format, time.localtime(ptime + 100000)))


def randomDate(start, end, prop):

    return (rangeTimeProp(start, end, '%Y-%m-%d', prop))

#print randomDate("1/1/2016", "09/1/2016", random.random())

def generateRandomTweets():

    for x in range(0, 10):
        dateRange = randomDate("2016-01-01", "2016-01-09", random.random())
        argv = ["--since", dateRange[0], "--until", dateRange[1], "--maxtweets", '1000']
        tweetQuery(argv, "tweetsForStopWord" + dateRange[0] + ".csv")
        print "Testing"
        print x

# generateRandomTweets()
def loadStopFiles():
    for stopFileName in listOfStopFileNames:
        with open('featureExtraction/stopWord/' + stopFileName, "rb") as stopFile:
            for line in stopFile:
                line = line[:-1]
                listOfStopWords.append(line)



def loadTweetFiles():
    tweetArray = []

    for filename in os.listdir('data'):
        with open('data/' + filename, "rb") as csvfile:
            tweets = csv.reader(csvfile, delimiter=";", quotechar="|")
            for row in tweets:
                tweetArray.append(row[4].lower())
    return tweetArray


def generateStopWords():
    loadStopFiles()
    tweetArray = []
    wordCount = collections.defaultdict(int)
    for filename in os.listdir(stopWordGenerationDir):
        with open(stopWordGenerationDir +'/' + filename, "rb") as csvfile:
            tweets = csv.reader(csvfile, delimiter=";", quotechar="|")
            for row in tweets:
                tweet = row[4].lower()
                tweetArray = tweet.split()
                for word in tweetArray:
                    print word
                    word = removePunctuation(word)
                    if word in listOfStopWords:
                        continue
                    wordCount[word] += 1
        

    sortedWordCount = sorted(wordCount.items(), key=operator.itemgetter(1), reverse=True)
    print sortedWordCount
    for index in range(38):
        listOfStopWords.append(sortedWordCount[index][0])

    text_file = open("finalStopWords.txt", "w")
    for word in listOfStopWords:
        text_file.write(word + "\n")
    text_file.close()

# ============================================================ ACTUAL STOP WORD REMOVAL

modifiedTweets = []

def removeStopWords():
    loadStopFiles()
    tweetArray = loadTweetFiles()
    modifiedTweetArray = []
    for tweet in tweetArray:
        tweet = removePunctuation(tweet)
        modifiedTweet = tweet.split(" ")
        for word in listOfStopWords:
            if word in modifiedTweet:
                modifiedTweet.remove(word)

        modifiedTweetArray.append(modifiedTweet)

    text_file = open("featureExtraction/stopWord/stoppedTweets.txt", "w")
    for tweet in modifiedTweetArray:
        text_file.write(str(tweet) + '\n')

    text_file.close()

# generateStopWords()
removeStopWords()
#print tweetArray


#print listOfStopWords
