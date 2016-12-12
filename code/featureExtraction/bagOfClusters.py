import gensim
import numpy 
import collections
import operator
import sys

# Load Google's pre-trained Word2Vec model.
# model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# print 'loaded'  
# model.init_sims(replace=True)

# x = model.most_similar(positive=['protest'])
# y = model['police'] - model['cop']
# d = numpy.linalg.norm(y)
# z = model['peace'] - model['violence']
# most_similar = model.most_similar(positive=['peace'], negative=['violence'])
# h = numpy.linalg.norm(z)

# la = model['gym'] - model['street']
# laz = numpy.linalg.norm(la)
# print 'similar to protest, ', x
# print 'raw numpy array difference between police and cop, ', y
# print 'normed difference between police and cop ', d
# print 'raw numpy diff between peace and violence ', z
# print 'normed difference between peace and violence ', h
# print 'most similar to peace, negative is violence ', most_similar
# print 'raw difference between gym and street ', la
# print 'normed difference between gym and street ', laz
# print 'we dun'

class Word2Vectorizer:
	def __init__(self):
		print "Loading model..."
		self.trainWord2Vec('GoogleNews-vectors-negative300.bin')

	def trainWord2Vec(self, trainingFile):
		self.model = gensim.models.Word2Vec.load_word2vec_format(trainingFile, binary=True)
		self.model.init_sims(replace=True)

	def supplyWordFrequencies(self, wordDict):
		self.wordDict = wordDict

	def getVecCountDictionary(self):
		self.vecDict = {}
		for word,count in self.wordDict:
			if word in self.model:
				self.vecDict[word] = (self.model[word], count)
		return self.vecDict

class BagOfClusters:
	def __init__(self, threshold = 30):
		self.bocFilename = "BoC.txt"
		self.word2Vectorizer = Word2Vectorizer()
		self.boc = []

	def generateBag(self, trainPool0, trainPool1):
		# get word counts
		wordCount0 = self.getWordCount(trainPool0)
		wordCount1 = self.getWordCount(trainPool1)

		# background subtraction
		wordDifference = self.backgroundSubtract(wordCount0, wordCount1)

		# converting words to vectors
		sortedWordCount = sorted(wordDifference.iteritems(), key=operator.itemgetter(1), reverse=True)

		self.word2Vectorizer.supplyWordFrequencies(sortedWordCount)
		word2VecDict = self.word2Vectorizer.getVecCountDictionary()

		print word2VecDict

		text_file = open(self.bocFilename, "w")
		for word in word2VecDict:
		    vec, count = word2VecDict[word]
		    text_file.write(word + ', ' + str(count) + ', ' + str(vec) + '\n')
		text_file.close()

		# clustering word vectors to get our bag of clusters
		clusters = self.cluster(word2VecDict)

	def getWordCount(self, wordPool):
		wordCount = collections.defaultdict(int)

		# creating word count bag
		for setTweets in wordPool:
		    for tweet in setTweets:
		        for word in tweet:
		            wordCount[word] += 1
		return wordCount

	def backgroundSubtract(self, wordCount0, wordCount1):
		""" Returns label 1 word counts minus label 0 word count """
		wordDifference = collections.defaultdict(int)
		for word in wordCount1:
		    if word in wordCount0:
		        wordDifference[word] = wordCount1[word] - wordCount0[word]
		    else:
		        wordDifference[word] = wordCount1[word]
		return wordDifference

	def cluster(self, word2VecDict):
		""" Gives us the most optimal high frequency clusters """

		# word2VecDict = dictionary mapping word to tuple with (vector, count)
		k = 100

		# initialize to random clusters
				# clusters = [numpy.zeros(52, 6)] * k
		cluster_center_keys = [ word2VecDict.keys()[i] for i in random.sample(xrange(len(word2VecDict)), k) ]

		# the values in our dict are (vec, count). Get the vecs
		cluster_centers = [ word2VecDict[word][0] for word in cluster_center_keys]
		word_assignments = [0] * len(word2VecDict.keys())

		# # assign each word vector to a cluster
		# # loop through word vectors
		# for word, (vec, count) in word2VecDict:
		# 	min_diff = "Inf"
		# 	min_point = 0
		# 	for cluster_center in cluster_centers:
		# 		diff = numpy.linalg.norm(cluster_center - vec)
				
		# 		if diff < min_diff:

		for word in word2VecDict:
			vec, count = word2VecDict[word]

	def saveBoC(self):
		millis = int(round(time.time() * 1000))
		boc_file = "BoC" + str(millis)

		print "Saving BoC to file..."
		bocFile = open("models/BoC/" + boc_file , 'w')
		for item in self.boc:
		    bocFile.write("%s\n" % item)
		print len(self.boc)