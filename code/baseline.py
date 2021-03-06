import sklearn
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import pandas as pd


#auth = tweedtype=str, delimiter=';', usecols=4)
tweetArray = [];
with open("output_got.csv", "rb") as csvfile:
    tweets = csv.reader(csvfile, delimiter=";", quotechar="|");
    for row in tweets:
        tweetArray.append(row[4]);

# for label 0 data
tweetArrayLabel0 = [];
with open("GetOldTweets-python/output_got1.csv", "rb") as csvfile:
    tweets = csv.reader(csvfile, delimiter=";", quotechar="|");
    for row in tweets:
        tweetArrayLabel0.append(row[4]);

# for test data
tweetArrayTest = [];
with open("test3.csv", "rb") as csvfile:
    tweets = csv.reader(csvfile, delimiter=";", quotechar="|");
    for row in tweets:
        tweetArrayTest.append(row[4]);


v = DictVectorizer(sparse=True);
#tweetMaps = [];
#for tweet in tweetArray:
#    newDict = {};
#    words = tweet.split();
#    for word in words:
#        newDict[word] = 1;
#

# split dataset
tweetDataSplit = []
tempsplitbuffer = []
for i in range(0, len(tweetArray)):
    tempsplitbuffer.append(tweetArray[i])
    if (len(tempsplitbuffer) >= 1000):
        tweetDataSplit.append(list(tempsplitbuffer))
        tempsplitbuffer = []

# split label 0 dataset
tweetDataSplitLabel0 = []
tempsplitbuffer = []
for i in range(0, len(tweetArrayLabel0)):
    tempsplitbuffer.append(tweetArrayLabel0[i])
    if (len(tempsplitbuffer) >= 1000):
        tweetDataSplitLabel0.append(list(tempsplitbuffer))
        tempsplitbuffer = []

# split test dataset
tweetDataSplitTest = []
tempsplitbuffer = []
for i in range(0, len(tweetArrayTest)):
    tempsplitbuffer.append(tweetArrayTest[i])

    if (len(tempsplitbuffer) >= 100):
        tweetDataSplitTest.append(list(tempsplitbuffer))
        tempsplitbuffer = []

# extract word presences

# loop through the data subsets
targetNames =  {}
featuresArray = []

for tweetDataSubset in tweetDataSplit:
    # loop through each tweet in the data subset and make a word presence array
    for tweet in tweetDataSubset:
        wordsInTweet = tweet.split()
        for word in wordsInTweet:
            targetNames[word] = 0

# loop through the label 0 dataset
for tweetDataSubset in tweetDataSplitLabel0:
    # loop through each tweet in the data subset and make a word presence array
    for tweet in tweetDataSubset:
        wordsInTweet = tweet.split()
        for word in wordsInTweet:
            targetNames[word] = 0

featuresArray = []
for tweetDataSubset in tweetDataSplit:
    feature = dict(targetNames)
    # loop through each tweet in the data subset and make a word presence array
    for tweet in tweetDataSubset:
        wordsInTweet = tweet.split()
        for word in wordsInTweet:
            if word in targetNames:
                feature[word] = 1
    featureArray = np.array(list(feature.values()))
    featuresArray.append(featureArray)

for tweetDataSubset in tweetDataSplitLabel0:
    feature = dict(targetNames)
    # loop through each tweet in the data subset and make a word presence array
    for tweet in tweetDataSubset:
        wordsInTweet = tweet.split()
        for word in wordsInTweet:
            if word in targetNames:
                feature[word] = 1
    print len(feature.values())
    featureArray = np.array(list(feature.values()))
    featuresArray.append(featureArray)

testsArray = []
for tweetDataSubset in tweetDataSplitTest:
    test = dict(targetNames)
    # loop through each tweet in the data subset and make a word presence array
    for tweet in tweetDataSubset:
        wordsInTweet = tweet.split()
        for word in wordsInTweet:
            if word in targetNames:
                test[word] = 1
    print len(test.values())
    testArray = np.array(list(test.values()))
    testsArray.append(testArray)

labels = [1 for x in range(10)]
labels0 = [0 for x in range(10)]
labels.extend(labels0)
clf = svm.SVC(gamma=0.001, C=100.)

labelArray = np.array(labels)
clf.fit(np.array(featuresArray), labelArray)

# here's the fun part. PREDICT!
prediction = clf.predict(np.array(testsArray))
