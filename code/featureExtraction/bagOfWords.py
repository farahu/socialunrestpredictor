import collections
import operator
import os

class BagOfWords:
    def __init__(self, threshold = 30):
        self.bogFilename = "wordBagDifference.txt"
        self.BOG_THRESHOLD = threshold
        self.bog = []

    def getBagOfWords(self):
        if len(self.bog) == 0:
            # generate and then return
            self.generateBagOfWords()
        return self.bog

    def generateBag(self, trainPool0, trainPool1):
        """ trainPool is a set of sets of words from social unrest data"""

        #self.bog = ["protest", "violent", "protesters", "night", "gas", "peaceful", "media", "tear", "protests", "cops", "crowd", "justice"]
        wordCount0 = collections.defaultdict(int)
        wordCount1 = collections.defaultdict(int)
        wordDifference = collections.defaultdict(int)

        for setTweets in trainPool0:
            for tweet in setTweets:
                for word in tweet:
                    wordCount0[word] += 1

        for setTweets in trainPool1:
            for tweet in setTweets:
                for word in tweet:
                    wordCount1[word] += 1

        for word in wordCount1:
            if word in wordCount0:
                wordDifference[word] = wordCount1[word] - wordCount0[word]
            else:
                wordDifference[word] = wordCount1[word]




        sortedWordCount = sorted(wordDifference.iteritems(), key=operator.itemgetter(1), reverse=True)
        text_file = open(self.bogFilename, "w")
        for word, count in sortedWordCount:
            text_file.write(word + ', ' + str(count) + '\n')
        text_file.close()

        # loop through till the word count hits our threshold. This will be our dictionary
        for word,count in sortedWordCount:
            if count < self.BOG_THRESHOLD:
                break
            self.bog.append(word)