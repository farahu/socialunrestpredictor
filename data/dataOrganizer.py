import collections
import operator
import os
import csv
import random

import sys
sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/featureExtraction")
from bagOfWords import BagOfWords

class DataOrganizer:
    def __init__(self):
        self.NUM_TWEETS_IN_SET = 100

    def organizeTrain(self, dataFileDir):
        # get all tweets from all train collections
        tweetArray0, tweetArray1 = self.readDataTrain(dataFileDir)

        # split in subsets
        setsOfTweets0 = self.randomSplit(tweetArray0)
        setsOfTweets1 = self.randomSplit(tweetArray1)

        return setsOfTweets0, setsOfTweets1

    def organizeTest(self, dataFileDir):
        # get all tweets from all train collections
        collectionArray = self.readDataTest(dataFileDir)
        setsOfTweets = [];
        # split in subsets
        for collection in collectionArray:
            for newSet in self.testSplit(collection):
                setsOfTweets.append(newSet);

        return setsOfTweets

    def testSplit(self, tweetArray):
        setOfSets = [];
        curSet = [];
        for x in range(0, len(tweetArray)):
            if len(curSet) == self.NUM_TWEETS_IN_SET:
                setOfSets.append(curSet)
                curSet = []
            curSet.append(tweetArray[x])

        return setOfSets

    def randomSplit(self,tweetArray):
        random.shuffle(tweetArray)

        setOfSets = []
        curSet = []
        for tweet in tweetArray:
            # put up to 100 in the one set
            curSet.append(tweet)

            if len(curSet) >= self.NUM_TWEETS_IN_SET-1:
                setOfSets.append(curSet)
                curSet = []
        return setOfSets

    def readDataTest(self, dataFileDir):
        """ Return array of tweets as strings"""

        # read in all csv files in data
        collectionArray = [];
        for fileName in os.listdir(dataFileDir):
            with open(dataFileDir + fileName, "rb") as csvfile:
                if '.csv' not in fileName:
                    continue

                tweets = csv.reader(csvfile, delimiter=";", quotechar="|")
                collection = [];
                for row in tweets:
                    collection.append(row[4].lower())
                collectionArray.append(collection)
                

        return collectionArray
        
    def readDataTrain(self, dataFileDir):

        """ Return array of tweets labeled 0 and labeled 1"""

        # read in all csv files in data
        tweetArray0 = []
        tweetArray1 = []
        for fileName in os.listdir(dataFileDir):
            with open(dataFileDir + fileName, "rb") as csvfile:
                if '.csv' not in fileName:
                    continue
                tweets = csv.reader(csvfile, delimiter=";", quotechar="|")
                for row in tweets:
                    if '1' in fileName:
                        tweetArray1.append(row[4].lower())
                    if '0' in fileName:
                        tweetArray0.append(row[4].lower())

        return (tweetArray0, tweetArray1)
