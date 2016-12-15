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

        self.VALIDATION_RATIO = .4

        # keeps track of how many sets should be labled 0 and how many labled 1 for accuracy measurement
        self.numTest0 = 0
        self.numTest1 = 0

    # def organizeTrainWithValidation(self, dataFileDir, resample):
    #     # get all tweets from all train collections

    #     # split in subsets
    #     # if(resample):
    #     trainArray0, trainArray1, validationArray0, validationArray1 = \
    #         self.readTrainWithValidationData(dataFileDir)


    #     setsOfTweets0 = self.randomSplit(trainArray0)
    #     setsOfTweets1 = self.randomSplit(trainArray1)
    #     setsOfTweets0Validation = self.randomSplit(validationArray0)
    #     setsOfTweets1Validation = self.randomSplit(validationArray1)

    #     self.saveTrainAndValidationSets(setsOfTweets0, setsOfTweets1, setsOfTweets0Validation, \
    #         setsOfTweets1Validation, dataFileDir)
    #     # else:
    #     #     setsOfTweets0, setsOfTweets1, setsOfTweets0Validation, setsOfTweets1Validation = \
    #     #         loadTrainAndValidationSets();


    #     return setsOfTweets0, setsOfTweets1, setsOfTweets0Validation, setsOfTweets1Validation

    def organizeTrain(self, dataFileDir):
        trainArray0, trainArray1 = self.readDataTrain(dataFileDir)


        setsOfTweets0 = self.randomSplit(trainArray0)
        setsOfTweets1 = self.randomSplit(trainArray1)

        return setsOfTweets0, setsOfTweets1
    # def loadTrainAndValidationSets(self):
    #     """ Reads from the train and validation folders and returns the sets """


    def saveTrainAndValidationSets(self, train0, train1, validate0, validate1, dataFileDir):
        dataFileTrainDir = dataFileDir + "/train/"
        dataFileValidateDir = dataFileDir + "/validate/"
        for fileName in os.listdir(dataFileTrainDir):
            os.remove(dataFileTrainDir + fileName)
        for fileName in os.listdir(dataFileValidateDir):
            os.remove(dataFileValidateDir + fileName)

        fileNamesDir = [dataFileTrainDir + "train0", dataFileTrainDir + "train1", \
           dataFileValidateDir + "validate0",dataFileValidateDir + "validate1"]

        train0File = open(fileNamesDir[0], 'w')

        for tweetSet in train0:
            for tweet in tweetSet:
                train0File.write("%s\n" % tweet)

        train1File = open(fileNamesDir[1], 'w')
        for tweetSet in train1:
            for tweet in tweetSet:
                train1File.write("%s\n" % tweet)

        validate0File = open(fileNamesDir[2], 'w')
        for tweetSet in validate0:
            for tweet in tweetSet:
                validate0File.write("%s\n" % tweet)

        validate1File = open(fileNamesDir[3], 'w')
        for tweetSet in validate1:
            for tweet in tweetSet:
                validate1File.write("%s\n" % tweet)


    def organizeTest(self, dataFileDir):
        # get all tweets from all train collections
        collections = self.readDataTest(dataFileDir)
        setsOfTweets = []

        # split in subsets. Also create a label vector to return for testing
        labels = []
        for collection, label in collections:
            for newSet in self.testSplit(collection):
                setsOfTweets.append(newSet)
                labels.append(label)

        return setsOfTweets, labels

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
        """ Return array of arrays of tweets as strings. Also counts number of 
        label 0 and label 1s in data set"""

        # save each collection of tweets with its label
        collectionsWithLabel = []
        for fileName in os.listdir(dataFileDir):
            with open(dataFileDir + fileName, "rb") as csvfile:
                if '.csv' not in fileName:
                    continue

                tweets = csv.reader(csvfile, delimiter=";", quotechar="|")

                # read in the collection
                collection = []
                for row in tweets:
                    collection.append(row[4].lower())

                # assign the label
                label = 1 if '1' in fileName else 0

                print fileName
                print "Label " + str(label)
        

                # append to our collection of collections
                collectionsWithLabel.append((collection, label))

                # increment the test labels
                # self.incrementTestLabels(fileName, len(collection))



        return collectionsWithLabel
        
    def readDataTrain(self, dataFileDir):

        """ Return array of tweets labeled 0 and labeled 1"""

        # read in all csv files in data
        tweetArray0 = []
        tweetArray1 = []
        for fileName in os.listdir(dataFileDir):
            if not os.path.isdir(dataFileDir + fileName):
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


    def readTrainWithValidationData(self, dataFileDir):

        """ Return array of tweets labeled 0 and labeled 1"""


        """
        #Pseudocode

        If user chooses to resample:
            -We'll clear the train and validate directories
            -For each collection:
                -Generate sets from tweets
                -Store generated sets in an array (distingusihed by 0 1)
                Ex: [Event1UnrestSets, Event2UnrestSets, Event1RestSets, GeneralRestSets]
                -


    

        """

        # read in all csv files in data
        tweetArray0 = []
        tweetArray1 = []
        for fileName in os.listdir(dataFileDir):
            if not os.path.isdir(dataFileDir + fileName):
                with open(dataFileDir + fileName, "rb") as csvfile:
                    if '.csv' not in fileName:
                        continue
                    tweets = csv.reader(csvfile, delimiter=";", quotechar="|")
                    for row in tweets:
                        if '1' in fileName:
                            tweetArray1.append(row[4].lower())
                        if '0' in fileName:
                            tweetArray0.append(row[4].lower())


        validationArray0 = []
        validationArray1 = []


        totalValidationSamples0 = len(tweetArray0) * self.VALIDATION_RATIO

        for x in range(0, int(totalValidationSamples0)):
            randomIndex = random.randint(0, len(tweetArray0)-1)
            validationArray0.append(tweetArray0[randomIndex])
            tweetArray0.remove(tweetArray0[randomIndex])


        totalValidationSamples1 = len(tweetArray1) * self.VALIDATION_RATIO
        for x in range(0, int(totalValidationSamples1)):
            randomIndex = random.randint(0, len(tweetArray1)-1)
            validationArray1.append(tweetArray1[randomIndex])
            tweetArray1.remove(tweetArray1[randomIndex])



        return tweetArray0, tweetArray1, validationArray0, validationArray1



    # def splitToValidation(self, dataFileDir):
    #     """ Splits our training data into validation and training 
    #     and saves both in the training and validation folders """
    #     fileNames = os.listdir(dataFileDir)
    #     print fileNames


    def incrementTestLabels(self, fileName, numTweets):
        if '1' in fileName:
            self.numTest1 += numTweets / self.NUM_TWEETS_IN_SET
        else:
            self.numTest0 += numTweets / self.NUM_TWEETS_IN_SET
