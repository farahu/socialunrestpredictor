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
	def __init__(self, threshold = 30, useOldWord2VectDict = True):
		self.bocFilename = "BoC.txt"
		self.word2VecDictFileName = "Word2VecDict.txt"
		self.word2Vectorizer = None
		self.boc = ()
		self.useOldDict = useOldWord2VectDict

	def bagTweets(self, *args):
		""" Take in a setOfSets of tweets. Return feature set for each setOfSets """

		X_all = []
		# run through each setOfSets which corresponds to some label
		for setsOfTweets in args:
			# print arg
			X_for_label = []
			# run through each set of tweets
			for setOfTweets in setsOfTweets:
				# each feature will be the average of all word2Vec
				X_for_label.append(self.getFeatureForSet(setOfTweets))

			X_all.append(X_for_label)

		return tuple(X_all)

	def getFeatureForSet(self, setOfTweets):
		""" For a set of tweets returns the weighted average word2Vec vector """

		# get word counts for each word in all tweets
		wordFreq = collections.defaultdict(int)
		for tweet in setOfTweets:
			for word in tweet:
				wordFreq[word] += 1

		print wordFreq

		# we can use the same word vectors but BEWARE the counts are different
		word2VecDict = self.word2Vectorizer.getVecCountDictionary()



	def getWord2VecDict(self, sortedWordCount):
		""" From sorted word count gives a dict of {word -> count, vec (from word2vec)}"""

		word2VecDict = {} 

		# either load from file or load up the model
		if self.useOldDict:
			# load old dict from self.word2VecDictFile
			word2VecDictFile = open("models/BoC/" + self.word2VecDictFileName)
			print "Building word2VecDict from file..."
			for line in word2VecDictFile:
				# we get word, count, vec from our text file
				word, count, vec = line.rstrip('\n').split(',')
				count = int(count)

				# print "------------------------------------------------------"
				# print "Raw string " +  str(len(vec))

				# print "Step A" + str(vec)
				# reshape vec into a 52 by 6 matrix
				# Brace yourselves. Messy code is coming
				vec = vec.replace("[","")

				# print "Step B" + str(vec)

				vec = vec.replace("]","")

				# print "Step C" + str(vec)

				# delimit by space into a list
				# print vec
				# vec = vec.replace(" ", "")
				vec = vec.split(' ')

				for index, numString in enumerate(vec):
					numString = numString.replace(" ", "")
					vec[index] = numString

				# print "Step D" + str(vec)

				vec2 = []
				for numString in vec:
					if numString != '':
						vec2.append(numString)
				vec = vec2
				# print "Length of vec: " + str(len(vec2))

				for index, numString in enumerate(vec):
					vec[index] = float(numString);
				# print "Step E" + str(vec)


				# print "------------------------------------------------------"
				vec = numpy.array(vec)

				# print word
				# print count

				# update our word2vecDict
				word2VecDict[word] = (vec, count)
				# print "The word vector" + str(vec)
		else:
			if self.word2Vectorizer == None:
				self.word2Vectorizer = Word2Vectorizer()
			self.word2Vectorizer.supplyWordFrequencies(sortedWordCount)
			word2VecDict = self.word2Vectorizer.getVecCountDictionary()
			self.saveDict(word2VecDict, self.word2VecDictFileName)

		return word2VecDict

	def generateBag(self, trainPool0, trainPool1):
		""" Returns the fit k-means model and a set of cluster indices we should consider"""

		print "Generating bag of clusters"

		# get word counts
		wordCount0 = self.getWordCount(trainPool0)
		wordCount1 = self.getWordCount(trainPool1)

		# background subtraction
		wordDifference = self.backgroundSubtract(wordCount0, wordCount1)

		# converting words to vectors
		sortedWordCount = sorted(wordDifference.iteritems(), key=operator.itemgetter(1), reverse=True)

		word2VecDict = self.getWord2VecDict(sortedWordCount)

		print "Built word2VecDict..."

		# text_file = open(self.bocFilename, "w")
		# for word in word2VecDict:
		#     vec, count = word2VecDict[word]
		#     text_file.write(word + ', ' + str(count) + ', ' + str(vec) + '\n')
		# text_file.close()

		# generate clusters
		learned_kmeans = self.cluster(word2VecDict)

		# pick top clusters with at least 30 words in it
		top_cluster_indices = self.pickTopClusters(30, learned_kmeans, word2VecDict)

		# our bag of clusters are basically the top cluster indices
		self.boc = (learned_kmeans, top_cluster_indices)
		
		return (learned_kmeans, top_cluster_indices)

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

	def pickTopClusters(self, thresholdClusterFreq, learned_kmeans, word2vecDict):
		""" Returns the indices of the top n clusters 
		based on the threshold. If a test word does not have one of these indices, 
		it is thrown out """

		cluster_counts = [0] * learned_kmeans.n_clusters
		for word, vec_count in word2vecDict.iteritems():
			# reshape vec to be a single sample
			index = learned_kmeans.predict(numpy.asarray(vec_count[0]).reshape(1,-1))

			# add one to the number of words in this cluster. Bingo!
			cluster_counts[index] += 1

		print cluster_counts

		# get indices of clusters that meet threshold
		top_cluster_indices = set()
		for index, cluster_count in enumerate(cluster_counts):
			if cluster_count > thresholdClusterFreq:
				top_cluster_indices.add(index)

		return top_cluster_indices


	def cluster(self, word2VecDict):
		""" Gives the learned k-means from word2vec vectors"""

		if len(word2VecDict.keys()) == 0:
			print "No words in your word2vec dict"
			return


		K = 10#len(word2VecDict.keys())

		# --------------------------- Sklearn implementation ----------------#

		# Create numpy array out of all the vectors
		num_words = len(word2VecDict.keys())
		orig_word2vec_dim = word2VecDict[word2VecDict.keys()[0]][0].shape

		X_python = []

		word2VecDictValues = word2VecDict.values()
		for vec, count in word2VecDictValues:
			X_python.append(vec)
		X = numpy.asarray(X_python)

		print "Clustering our word vectors for Bag of Clusters"

		# Run K-means with different K and minimize our loss function
		k_means = KMeans(n_clusters=K, init='random', precompute_distances='auto')

		k_means.fit(X)

		print "K-means error: " + str(k_means.inertia_)

		return k_means

		# assignments = k_means.labels_
		# print "Length of assignments vector: " + str(len(assignments))
		# print "Should be " + str(num_words)

		# # get the maximum radius for each centroid
		# getRadiuses(assignments, word2VecDictValues)


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


	# def getRadius(self, assignments, word2VecDictValues):
	# 	""" Returns radius for each centroid in kmeans (radius by centroid index) """

	# 	# run through each word, vec, count combo
	# 	for 

	def saveDict(self, dictToSave, fileName):
		# millis = int(round(time.time() * 1000))
		fileName = fileName 
		# + str(millis)

		print "Saving " + fileName + " file..."
		file = open("models/BoC/" + fileName , 'w')
		for word, (vec, count) in dictToSave.iteritems():
			file.write(word + "," + str(count) + "," + numpy.array_str(vec, max_line_width=100000) + "\n")
		print len(dictToSave)