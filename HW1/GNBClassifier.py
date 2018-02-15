import csv, random, math
class GNB(object):
	def loadData(self, file):
		lines = csv.reader(open(file, "rb"))
		dataset = list(lines)
		for i in range(len(dataset)):
			dataset[i] = [float(x) for x in dataset[i]]

		return dataset

	def splitDataset(self, dataset):
		trainingSize = int(len(dataset) * 0.67)
		trainingSet = []
		testSet = list(dataset)
		while len(trainingSet) < trainingSize:
			i = random.randrange(len(testSet))
			trainingSet.append(testSet.pop(i))
		return [trainingSet, testSet]

	def splitDataSetUsingFractions(self, dataset):
		fractions = [.01, .02, .05, .1, .625, 1]

		result = []

		for fraction in fractions:
			OriginaltrainingSet, testSet = self.splitDataset(dataset)
			newTrainingSize = fraction * len(OriginaltrainingSet)
			newTrainingSet = list(OriginaltrainingSet)
			
			while len(newTrainingSet) > newTrainingSize:
				i = random.randrange(len(newTrainingSet))
				newTrainingSet.pop(i)
			result.append((newTrainingSet, testSet))
		return result


	def separateByLabels(self, dataset):
		labels = {}
		for i in range(len(dataset)):
			currRow = dataset[i]
			if (currRow[-1] not in labels):
				labels[currRow[-1]] = []
			labels[currRow[-1]].append(currRow)
		return labels


	def mu(self, nums):
		return sum(nums)/float(len(nums))

	def sigma(self, nums):
		avg = self.mu(nums)
		variance = sum([pow(x-avg,2) for x in nums])/float(len(nums)-1)
		return math.sqrt(variance)

	def summarizeDatasetByAttributes(self, dataset):
		summaryList = [(self.mu(attribute), self.sigma(attribute)) for attribute in zip(*dataset)]
		del summaryList[-1]
		return summaryList

	def summarizeDatasetByLabels(self, dataset):
		lables = self.separateByLabels(dataset)
		summaryDict = {}
		for label, value in lables.iteritems():
			summaryDict[label] = self.summarizeDatasetByAttributes(value)
		return summaryDict

	def calculateProbability(self, val, mu, sigma):
		exponent = math.exp(-(math.pow(val-mu,2)/(2*math.pow(sigma,2))))
		return (1 / (math.sqrt(2*math.pi) * sigma)) * exponent

	def ProbabilityPerClass(self, summaryDict, inputList):
		probabilities = {}
		for label, labelSummary in summaryDict.iteritems():
			probabilities[label] = 1
			for i in range(len(labelSummary)):
				mu, sigma = labelSummary[i]
				x = inputList[i]
				probabilities[label] *= self.calculateProbability(x, mu, sigma)
		return probabilities

	def predict(self, labelSummaries, inputList):
		probabilities = self.ProbabilityPerClass(labelSummaries, inputList)
		bestLabel, bestProb = None, -1
		for classValue, probability in probabilities.iteritems():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = classValue
		return bestLabel

	def getPredictions(self, labelSummaries, testSet):
		predictions = []
		for i in range(len(testSet)):
			result = self.predict(labelSummaries, testSet[i])
			predictions.append(result)
		return predictions

	def getAccuracy(self, testSet, predictions):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == predictions[x]:
				correct += 1
		return (correct/float(len(testSet))) * 100.0

if __name__ == "__main__":
	file = 'data_banknote_authentication.csv'
	obj = GNB()
	dataset = obj.loadData(file)
	accuracyList = [0,0,0,0,0,0]
	for k in range(5):
		data = obj.splitDataSetUsingFractions(dataset)
		j = 0
		for i in range(len(data)):
			trainingSet = data[i][0]
			testSet = data[i][1]
			# prepare model
			labelSummaries = obj.summarizeDatasetByLabels(trainingSet)
			# test model
			predictions = obj.getPredictions(labelSummaries, testSet)
			accuracy = obj.getAccuracy(testSet, predictions)

			#print j
			accuracyList[j] += accuracy
			j += 1
			#print('Accuracy: {0}%').format(accuracy)
	#print accuracyList
	for k in range(5):
		print accuracyList[k]/5
 


