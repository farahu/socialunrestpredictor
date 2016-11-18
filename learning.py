import collections
import operator
import os
import sys

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/featureExtraction")
from featureExtractor import FeatureExtractor

STOP_WORD_CUTOFF = 38
DICTIONARY_CUTOFF = 3887

def readData():
    """" Returns data as word frequency vector """

    # for now we already have data
    wordCount = collections.defaultdict(int)
    with open("featureExtraction/stopWord/stoppedTweets.txt") as f:
        for line in f:
            tempList = eval(line)
            for word in tempList:
                wordCount[word] += 1

    return wordCount

def learn():
    # take in data
    wordCount = readData()

    # use BoG to convert to frequency vector
    fe = FeatureExtractor()
    featureVector = fe.extractFeatureVectors(wordCount)    
    # call sk

learn()