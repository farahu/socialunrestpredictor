import gensim
import numpy 
import collections
import operator
import sys
import random
import time
import os
from sklearn.cluster import KMeans

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
	def __init__(self, threshold = 30, useOldWord2VectDict = False):
		self.bocFilename = "BoC.txt"
		self.word2VecDictFileName = "Word2VecDict.txt"
		self.word2Vectorizer = None
		self.boc = []
		self.useOldDict = useOldWord2VectDict

	def getWord2VecDict(self, sortedWordCount):
		""" From sorted word count gives a dict of {word -> count, vec (from word2vec)}"""
		print self.useOldDict

		word2VecDict = {} 

		# either load from file or load up the model
		if self.useOldDict:
			# load old dict from self.word2VecDictFile
			word2VecDictFile = open("models/BoC/" + self.word2VecDictFileName)
			for line in word2VecDictFile:
				# we get word, count, vec from our text file
				word, count, vec = line.rstrip('\n').split(',')
				count = int(count)
				print vec
				vec = numpy.matrix(vec)
				print word
				print count
				print "The word vector" + str(vec)
		else:
			if self.word2Vectorizer == None:
				self.word2Vectorizer = Word2Vectorizer()
			self.word2Vectorizer.supplyWordFrequencies(sortedWordCount)
			word2VecDict = self.word2Vectorizer.getVecCountDictionary()
			self.saveDict(word2VecDict, self.word2VecDictFileName)

		return word2VecDict

	def generateBag(self, trainPool0, trainPool1):
		print "Generating bag of clusters"

		# get word counts
		wordCount0 = self.getWordCount(trainPool0)
		wordCount1 = self.getWordCount(trainPool1)

		# background subtraction
		wordDifference = self.backgroundSubtract(wordCount0, wordCount1)

		# converting words to vectors
		sortedWordCount = sorted(wordDifference.iteritems(), key=operator.itemgetter(1), reverse=True)

		word2VecDict = self.getWord2VecDict(sortedWordCount)

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

		print "Num unique words: " + str(len(wordDifference))
		return wordDifference

	def cluster(self, word2VecDict):
		""" Gives us the most optimal high frequency clusters """

		if len(word2VecDict.keys()) == 0:
			print "No words in your word2vec dict"
			return

		print "Clustering our word vectors for Bag of Clusters"

		K = 1#len(word2VecDict.keys())

		# --------------------------- Sklearn implementation ----------------#

		# Create numpy array out of all the vectors
		num_words = len(word2VecDict.keys())
		orig_word2vec_dim = word2VecDict[word2VecDict.keys()[0]][0].shape

		X_python = []
		for vec, count in word2VecDict.values():
			X_python.append(vec)
		X = numpy.asarray(X_python)

		print "Loaded numpy array for K-means"

		# Run K-means with different K and minimize our loss function

		k_means = KMeans(n_clusters=K, init='random')
		k_means.fit(X)

		print "K-means error: " + str(k_means.inertia_)


		# --------------------------- Our own implementation ----------------#

		# word2VecDict = dictionary mapping word to tuple with (vector, count)

		# # reshape our arrays
		# orig_word2vec_dim = word2VecDict[word2VecDict.keys()[0]][0].shape


		# # pick random keys (words) whose vectors will be the initial clusters
		# cluster_center_keys = [ word2VecDict.keys()[i] for i in random.sample(xrange(len(word2VecDict)), k) ]

		# # Get the random vecs from the random keys. Make these our intial clusters
		# cluster_centers = [ word2VecDict[word][0] for word in cluster_center_keys]

		# print "Initial cluster centers"
		# print cluster_centers

		# # keep track of which cluster a word is assigned to. {cluster_index -> word1, word2}
		# cluster_word_dict = collections.defaultdict(list)


		# # assign each word vector to a cluster loop through word vectors
		# for word, (vec, count) in word2VecDict:
		# 	min_dist = "Inf"
		# 	min_dist_cluster_ind = 0

		# 	for index, cluster_center in enumerate(cluster_centers):
		# 		cluster_dist = numpy.linalg.norm(cluster_center - vec)
		# 		if cluster_dist < min_dist:
		# 			min_dist_cluster_ind = index
		# 			min_dist = cluster_dist

		# 	cluster_word_dict[min_dist_cluster_ind].append(vec) 

		# 	print "Word to cluster assignments" + word_cluster_assignments

		# # make our clusters the average of its corresponding word vectors
		# for index, cluster_center in enumerate(cluster_centers):
		# 	word_vecs = cluster_word_dict[index]

		# 	mean_vec = numpy.zeros(orig_word2vec_dim)

		# 	for vec in word_vecs:
		# 		mean_vec += vec
		# 	mean_vec /= len(word_vecs)

		# 	cluster_centers[index] = mean_vec

		# print "One iteration cluster centers"
		# print cluster_centers


	def saveDict(self, dictToSave, fileName):
		# millis = int(round(time.time() * 1000))
		fileName = fileName 
		# + str(millis)

		print "Saving " + fileName + " file..."
		file = open("models/BoC/" + fileName , 'w')
		for word, (vec, count) in dictToSave.iteritems():
		    file.write(word + "," + str(count) + "," + numpy.array_str(vec, max_line_width=100000) + "\n")
		print len(dictToSave)