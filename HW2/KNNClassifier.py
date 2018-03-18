import csv
import math
import operator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance

class ProcessData(object):
	def loadData(self, training_file, test_file):
		trainingSet = []
		with open(training_file, 'rb') as csvfile:
			lines = csv.reader(csvfile)
			dataset = list(lines)
			for i in range(6000):
				trainingSet.append(dataset[i])

		testSet = []
		with open(test_file, 'rb') as csvfile:
			lines = csv.reader(csvfile)
			dataset = list(lines)
			for i in range(1000):
				testSet.append(dataset[i])

		return trainingSet, testSet

class KNN(object):
	def euclideanDistance(self, example1, example2):
		A = example1[1:]
		B = example2[1:]
		
		dist = np.sqrt(np.sum((A-B)**2)) 
		return dist

	def getKNeighbours(self, trainingSet, testExample, KList):
		distances = []

		for i in range(len(trainingSet)):
			currDist = self.euclideanDistance(trainingSet[i], testExample)
			distances.append((trainingSet[i], currDist))

		sortedDict = sorted(distances, key=lambda x: x[1])

		topK = []
		for k in KList:
			curr = []
			for i in range(k):
				curr.append(sortedDict[i][0])
			topK.append(curr)
		return topK

	def getPrediction(self, topKList):
		result = []
		for topK in topKList:
			labelsDict = {}
			for item in topK:
				currLabel = item[0]
				try:
					labelsDict[currLabel] += 1
				except KeyError:
					labelsDict[currLabel] = 1

			sortedLabelsDict = sorted(labelsDict.iteritems(), key=operator.itemgetter(1), reverse=True)

			result.append(sortedLabelsDict[0][0])
		return result

	def getAccuracy(self, testSet, predictions):
		accuracyList = [0] * 11
		correctLabels = [0] * 11
		incorrectLabels = [0] * 11
		for i in range(len(testSet)):
			for j in range(len(predictions[i])):
				if testSet[i][0] == predictions[i][j]:
					correctLabels[j] += 1

		for i in range(11):
			incorrectLabels[i] = len(testSet) - correctLabels[i]
			accuracyList[i] = correctLabels[i]/float(len(testSet))

		return accuracyList, incorrectLabels

class PlotData(object):
	def __init__(self):
		plt.xlabel("K (No of Neighbours)")
		plt.ylabel("Error Rate")

	def plotCurve(self, data1, data2, op):
		plt.plot(data1, data2, op)
	
	def showCurve(self):
		plt.show()

		

if __name__=="__main__":
	training_file = 'Dataset/mnist_train.csv'
	test_file = 'Dataset/mnist_test.csv'

	KList = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

	pd = ProcessData()
	trainingSet, testSet = pd.loadData(training_file, test_file)

	trainingSet = np.array(trainingSet)
	trainingSet = trainingSet.astype(float)

	testSet = np.array(testSet)
	testSet = testSet.astype(float)

	knn = KNN()
	predictions = []
	for example in testSet:
		topKNeighbour = knn.getKNeighbours(trainingSet, example, KList)
		predictedLabel = knn.getPrediction(topKNeighbour)
		predictions.append(predictedLabel)

	accuracyList, incorrectLabels = knn.getAccuracy(testSet, predictions)


	TestErrorList = []

	for accuracy in accuracyList:
		TestErrorList.append(1.0 - accuracy)


	for i in range(len(KList)):
		print "For K = ", KList[i]
		print "Test Error: ", TestErrorList[i]
		print "No of Incorrect Labels: ", incorrectLabels[i]
		print "---------------------------------------------"
	#print "Test Errors: ", TestErrorList
	#print "No of Incorrect Labels: ", incorrectLabels
	plot = PlotData()
	plot.plotCurve(KList, TestErrorList,'ro-')
	plot.showCurve()
