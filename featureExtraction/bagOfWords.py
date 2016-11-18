import collections
import operator
import os

class BagOfWords:
    def __init__(self, threshold = 3):
        self.bogFilename = "featureExtraction/wordBag.txt"
        self.BOG_THRESHOLD = threshold
        self.BOG = []

    def getBagOfWords(self):
        if len(self.BOG) == 0:
            # generate and then return
            self.generateBagOfWords()
        return self.BOG

    def generateBagOfWords(self):
        wordCount = collections.defaultdict(int)

        with open("featureExtraction/stopWord/stoppedTweets.txt") as f:
            for line in f:
                tempList = eval(line)
                for word in tempList:
                    wordCount[word] += 1

        sortedWordCount = sorted(wordCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        text_file = open("wordBag.txt", "w")
        for word, count in sortedWordCount:
            text_file.write(word + ', ' + str(count) + '\n')
        text_file.close()

        # loop through till the word count hits our threshold. This will be our dictionary
        for word,count in sortedWordCount:
            if count < self.BOG_THRESHOLD:
                break
            self.BOG.append(word)