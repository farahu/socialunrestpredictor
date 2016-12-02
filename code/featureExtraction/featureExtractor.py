import collections
import operator
import os

import sys
sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/code/featureExtraction")
from bagOfWords import BagOfWords
sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/code/featureExtraction/stopWord")
from stopWordRemoval import removePunctuation
from stopWordRemoval import StopWordRemover

class FeatureExtractor:

    def bagTweets(self, setOfTweetsWordCount):
        """" takes in a word count dictionary {word, wordCount} 
        for a set of tweets and outputs a frequency vector """

        # we generate one frequency vector for reach set
        freqVector = []


        # loop through our bag of words model
        for ithWord in self.bog.bog:
            freqVector.append(setOfTweetsWordCount[ithWord])
        return freqVector

    def getWordCountDict(self, setOfTweets):
        """ takes a set of tweets and returns a dictionary of word -> wordCount"""
        wordCount = collections.defaultdict(int)

        for tweet in setOfTweets:
            for word in tweet:
                wordCount[word] += 1

        return wordCount

    def removePuncStopTokenize(self, setsOfSets, stopWordRemover):
        """ removes punctuation, removes stop words and tokenizes tweets
        into word arrays"""

        newSets = []
        # loop through each set 
        for curSet in setsOfSets:
            stoppedSet = []
            for i, tweet in enumerate(curSet):
                # remove punctuation
                updatedTweet = removePunctuation(tweet) 
                stoppedSet.append(stopWordRemover.removeStopWords(updatedTweet))

            newSets.append(stoppedSet)

        return newSets

    def extractTrainFeatureVectors(self, allTrainData):
        """ takes in 2 sets of tweets. One for train 0 and for train 1
         and turns each set in both of these pool into a feature vector""" 

        setsOfSets0, setsOfSets1 = allTrainData

        stopWordRemover = StopWordRemover()

        # prune our sets of sets of tweets and tokenize
        setsOfSets0 = self.removePuncStopTokenize(setsOfSets0, stopWordRemover)
        setsOfSets1 = self.removePuncStopTokenize(setsOfSets1, stopWordRemover)

        # generate bag of words from the label 1 train pool
        self.bog = BagOfWords()
        self.bog.generateBag(setsOfSets0, setsOfSets1)


        # now we want to generate a feature vector for each set. Right now the last
        # step to generate these feature vecotrs is just bagging
        X0 = []
        for setOfTweets in setsOfSets0:
            # convert each set of tweets to a wordCount dictionary
            setWordCountDict = self.getWordCountDict(setOfTweets)

            # bag the set of tweets through its wordCount dictionary
            X0.append(self.bagTweets(setWordCountDict))


        # do the same for X1
        X1 = []
        for setOfTweets in setsOfSets1:
            # convert each set of tweets to a wordCount dictionary
            setWordCountDict = self.getWordCountDict(setOfTweets)

            # bag the set of tweets through its wordCount dictionary
            X1.append(self.bagTweets(setWordCountDict))

        return X0, X1

    def extractTestFeatureVectors(self, allTestData):
        stopWordRemover = StopWordRemover()

        # prune our sets of sets of tweets and tokenize
        allTestData = self.removePuncStopTokenize(allTestData, stopWordRemover)

        # now we want to generate a feature vector for each set. Right now the last
        # step to generate these feature vecotrs is just bagging
        testFeatureVectors = []
        for setOfTweets in allTestData:
            # convert each set of tweets to a wordCount dictionary
            setWordCountDict = self.getWordCountDict(setOfTweets)

            # bag the set of tweets through its wordCount dictionary
            testFeatureVectors.append(self.bagTweets(setWordCountDict))

        return testFeatureVectors








