import collections
import operator
import os

import sys
sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/featureExtraction")
from bagOfWords import BagOfWords

class FeatureExtractor:
    def bagTweets(self, wordCount):
        """" wordCount : Dictionary of {word, count} """

        bog = BagOfWords()
        baggedTweets = []
        # loop through our bag of words model
        for word in bog.getBagOfWords():
            baggedTweets.append(wordCount[word])

        return baggedTweets

    def extractFeatureVectors(self, wordCount):
        return self.bagTweets(wordCount)
