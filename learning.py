import collections
import operator
import os
import sys
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/featureExtraction")
from featureExtractor import FeatureExtractor

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/data")
from dataOrganizer import DataOrganizer

STOP_WORD_CUTOFF = 38
DICTIONARY_CUTOFF = 3887

# def readData():
#     """" Returns data as word frequency vector """

#     # for now we already have data
#     wordCount = collections.defaultdict(int)
#     with open("featureExtraction/stopWord/stoppedTweets.txt") as f:
#         for line in f:
#             tempList = eval(line)
#             for word in tempList:
#                 wordCount[word] += 1

#     return wordCount

def convertListOfFeatures(listOfListsOfFeatures):
    """ takes of a list OF a list of feature vectors and returns a feature numpy array"""
    # get the size of each of our vectors to create our np array

    height = 0
    for listOfFeatureVectors in listOfListsOfFeatures:
        height += len(listOfFeatureVectors)

    featureArray = np.zeros((height, len(listOfListsOfFeatures[0][0])))

    # now loop through our lists and put them into this feature array
    rowIndex = 0
    for listOfFeatureVectors in listOfListsOfFeatures:
        for featureVector in listOfFeatureVectors:
            featureArray[rowIndex,:] = np.array(featureVector)
            rowIndex += 1

    return featureArray

def learn(X0, X1):
    """ Learns based on two feature vectors whose label is in the name """
    if len(X0) == 0:
        return

    Y0 = [0 for x0 in X0]
    Y1 = [1 for x1 in X1]
    labels = Y0 + Y1
    labelArray = np.array(labels)

    # we have to convert everything to np.arrays -.-
    featureArray = convertListOfFeatures([X0, X1])

    clf = svm.SVC(gamma=0.001, C=100)

    clf.fit(featureArray, labelArray)

    return clf

    # # here's the fun part. PREDICT!
    # prediction = clf.predict(np.array(testsArray))

def test(clf, testSet):
    """ takes in a list of feature vectors and a fit svm and then predicts """

    # convert to np array
    featureArray = convertListOfFeatures([testSet])

    print clf.predict(featureArray)

def main():
    """" Preprocesses, extracts, learns, tests"""

    # preprocessing
    do = DataOrganizer()

    # get sets of tweets as training data
    trainData0, trainData1 = do.organizeTrain("data/train/")
    
    # get sets of tweets of testing data
    testData = do.organizeTest("data/test/")

    # take in data
    # wordCount = readData()

    # use BoG to convert to frequency vector
    fe = FeatureExtractor()
    X0, X1 = fe.extractTrainFeatureVectors((trainData0, trainData1))

    testFeatures = fe.extractTestFeatureVectors((testData))

    clf = learn(X0, X1)

    test(clf, testFeatures)

    # call sk

main()