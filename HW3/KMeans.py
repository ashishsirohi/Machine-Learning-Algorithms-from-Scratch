import csv
import math
import operator
import matplotlib.pyplot as plt
import numpy as np

class ProcessData(object):
	def loadData(self, file):
		trainingSet = []
		with open(file, 'rb') as csvfile:
			lines = csv.reader(csvfile)
			dataset = list(lines)
			for line in dataset:
				trainingSet.append(line[1:10])

		return trainingSet

class KMeans(object):
	def __init__(self):
		pass

	def euclideanDistance(self, A, B):
		dist = np.sqrt(np.sum((A-B)**2)) 
		return dist

	def classifyStep(self, dataset, centroids):
		clusters = []
		for i in range(len(centroids)):
			clusters.append([])

		for datapoint in dataset:
			cluster = None
			minVal = float('inf')
			for i in range(len(centroids)):
				curr = self.euclideanDistance(datapoint, centroids[i])
				if curr < minVal:
					minVal = curr
					cluster = i

			clusters[cluster].append(datapoint)

		return clusters

	def recenterStep(self, clusters):
		new_centroid = []
		for cluster in clusters:
			new_centroid.append(np.mean(cluster, axis=0))
		
		return new_centroid


	def kMeansConvergeStep(self, dataset, centroids, k):
		i = 0
		tolerance = 0.0001
		while i < 1000:
			clusters = self.classifyStep(dataset, centroids)
			for c in clusters:
				if len(c) == 0:
					print clusters
					break
			new_centroids = self.recenterStep(clusters)

			diff = 0

			for i in range(len(new_centroids)):
				diff += self.euclideanDistance(centroids[i], new_centroids[i])

			#print diff
			error = diff/float(len(centroids))

			print error
			if error < tolerance:
				break
			else:
				centroids = list(new_centroids)

			i += 1

		return new_centroids, clusters

	def calculatePotentialFunction(self, centroids, clusters):
		potentialFn = 0
		for i in range(len(centroids)):
			curr = 0
			for j in range(len(clusters[i])):
				tmp = centroids[i] - clusters[i][j]
				curr += np.sum(tmp ** 2)

			potentialFn += curr

		return potentialFn


class PlotData(object):
	def __init__(self):
		plt.xlabel("K (No of Neighbours)")
		plt.ylabel("Error Rate")

	def plotCurve(self, data1, data2, op):
		plt.plot(data1, data2, op)
	
	def showCurve(self):
		plt.show()





if __name__ == "__main__":
	file = 'bc.txt'

	kValues = [2, 3, 4, 5, 6, 7, 8]

	pd = ProcessData()
	dataset = pd.loadData(file)
	dataset = np.array(dataset)
	dataset = dataset.astype(float)

	km = KMeans()

	potentialFn = []
	for k in kValues:
		intial_centroid_indexes = np.random.randint(0, len(dataset), size=k)
		initial_centroids = []
		for index in intial_centroid_indexes:
			initial_centroids.append(dataset[index])
		final_centroids, final_clusters = km.kMeansConvergeStep(dataset, initial_centroids, k)
		L_Value = km.calculatePotentialFunction(final_centroids, final_clusters)
		potentialFn.append(L_Value)

	print potentialFn
	plot = PlotData()
	plot.plotCurve(kValues, potentialFn,'ro-')
	plot.showCurve()




