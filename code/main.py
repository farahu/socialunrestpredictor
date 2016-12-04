import collections
import operator
import os
import sys
import numpy as np
import time
from plot import plot
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.externals import joblib

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/code/featureExtraction")
from featureExtractor import FeatureExtractor
from bagOfWords import BagOfWords

sys.path.insert(0, "/Users/tariq/Dev/School/socialunrestpredictor/code/")
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
    """ takes in a list of feature vectors and a fit svm and then returns predicted y labels """

    # convert to np array
    featureArray = convertListOfFeatures([testSet])

    return clf.predict(featureArray)

def main():
    """" Preprocesses, extracts, learns, tests"""

    # process flags
    do_retrain = False
    for arg in sys.argv[1:]:
        if ("--retrain" in arg):
            if ("yes" in arg):
                do_retrain = True

    # preprocessing
    do = DataOrganizer()

    # __________________________________ TRAINING ________________________ #

    # use BoG to convert to frequency vector
    fe = FeatureExtractor()

    clf = 0
    clf_file = ""

    # get the latest trained model
    filenames = os.listdir("models/")
    if len(filenames) > 0:
        clf_file = "models/" + filenames[-1]
    else:
        clf_file = None

    if do_retrain or not clf_file:
        # get sets of tweets as training data
        trainData0, trainData1 = do.organizeTrain("data/train/")

        # split training set into validation and training set
        # trainData0, trainData1, validation = do.splitIntoValidation()
        X0, X1 = fe.extractTrainFeatureVectors((trainData0, trainData1))
        clf = learn(X0, X1)

        millis = int(round(time.time() * 1000))
        clf_file = "trainedModel" + str(millis)
        print "Saving model to file..."

        joblib.dump(clf, "models/" + clf_file, compress = 1) 
    else:
        print "Using trained model and BoG..."
        fe.bog = BagOfWords()
        fe.bog.getLatestBoG()
        clf = joblib.load(clf_file) 
        
    # ____________________________________TESTING __________________________ #

    # get sets of tweets of testing data
    testData = do.organizeTest("data/test/")

    testFeatures = fe.extractTestFeatureVectors((testData))

    # in the future save and only relearn when needed   
    yPred = test(clf, testFeatures)

    # plot results
    
    # generate yActual. For now its manual
    numLabel1 = 100
    yActual = [1 for i in range(numLabel1)]
    yActual.extend([0 for i in range(len(yPred)-numLabel1)])

    print yActual
    print yPred

    
    plot(yActual, yPred, ["No Social Unrest", "Social Unrest"])

if __name__ == "__main__":
    main()