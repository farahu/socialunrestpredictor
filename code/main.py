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

# def validate(clf, validation0, validation1):
#     featureArray = convertListOfFeatures([validation0, validation1])
#     yPred = clf.predict(featureArray)

#     yActual = [0 for x in range(len(validation0))]
#     yActual.extend([1 for x in range(len(validation1))])

#     plot("Validation", yActual, yPred, ["No Social Unrest", "Social Unrest"])


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

def test(action, clf, testSet, testLabels):
    """ takes in a list of feature vectors and a fits svm and plots results
    Can be used for validation as well

    action: either "testing" or "validation" """

    # convert to np array
    featureArray = convertListOfFeatures([testSet])

    yPred = clf.predict(featureArray)

    # plot results    
    # # generate yActual. For now its manual
    # yActual = [0 for i in range(numTest0)]
    # yActual.extend([1 for i in range(numTest1)])

    printError(yPred, testLabels)
    
    plot(testLabels, yPred, ["No Social Unrest", "Social Unrest"], action)

def printError(yPred, testLabels):
    errorCount = 0.0
    for i, trueLabel in enumerate(yPred):
        if trueLabel != testLabels[i]:
            errorCount += 1.0

    print "Error is: " + str(errorCount/len(testLabels))

def main():
    """" Preprocesses, extracts, learns, tests"""

    # process flags
    do_retrain, do_rebuildValidation, do_test = False, False, False

    for arg in sys.argv[1:]:
        if ("--retrain" in arg):
            if ("yes" in arg):
                do_retrain = True
        if ("--rebuildValidation" in arg):
            if ("yes" in arg):
                do_rebuildValidation = True
        if ("--test" in arg):
            if ("yes" in arg):
                do_test = True

    # preprocessing
    do = DataOrganizer()

    # __________________________________ TRAINING ________________________ #

    # use BoG to convert to frequency vector

    fe = FeatureExtractor(FeatureExtractor.ModelType.BagOfWords)

    clf = 0
    clf_file = ""

    # get the latest trained model
    filenames = os.listdir("models/")
    if len(filenames) > 0:
        clf_file = "models/" + filenames[-1]
    else:
        clf_file = None

    # get sets of tweets as training data
    # trainData0, trainData1, validation0, validation1 \
    #     = do.organizeTrainWithValidation("data/trainValidate/", do_rebuildValidation)

    trainData0, trainData1 = do.organizeTrain("data/train/")

    if do_retrain or not clf_file:
        # split training set into validation and training set
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

    # we're either validating or testing based on the passed flag

    # ____________________________________VALIDATION__________________________#
    if not do_test:
        # feed in the validation sets as one set
        validationData = do.organizeTest("data/validation/")
        validationFeatures = fe.extractTestFeatureVectors(validationData)
        test("Validation", clf, validationFeatures, do.numTest0, do.numTest1)
    else:
        # ____________________________________TESTING _______________________ #

        # extract test features and test
        print "Using testing"
        testData, testLabels = do.organizeTest("data/test/")
        testFeatures = fe.extractTestFeatureVectors(testData) 
        test("Testing, Global Protests With Background Subtraction", clf, testFeatures, testLabels)

if __name__ == "__main__":
    main()