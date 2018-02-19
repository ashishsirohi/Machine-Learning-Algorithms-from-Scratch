import csv, random, math
import numpy as np
import matplotlib.pyplot as plt

class ProcessData(object):
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


class GNB(object):
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
		try:
			variance = sum([pow(x-avg,2) for x in nums])/float(len(nums)-1)
		except:
			variance = sum([pow(x-avg,2) for x in nums])/float(len(nums))
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

	def calculateLikelihood(self, val, mu, sigma):
		exponent = math.exp(-(math.pow(val-mu,2)/(2*math.pow(sigma,2))))
		return (1 / (math.sqrt(2*math.pi) * sigma)) * exponent

	def calculatePosterier(self, summaryDict, inputList):
		probabilities = {}
		for label, labelSummary in summaryDict.iteritems():
			probabilities[label] = 1
			for i in range(len(labelSummary)):
				mu, sigma = labelSummary[i]
				x = inputList[i]
				probabilities[label] *= self.calculateLikelihood(x, mu, sigma)
		return probabilities

	def predict(self, labelSummaries, inputList):
		probabilities = self.calculatePosterier(labelSummaries, inputList)
		bestLabel, bestProb = None, -1
		for label, probability in probabilities.iteritems():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = label
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


class LogisticRegression(object):
	def __init__(self, dataset):
		self.coeff = np.asarray([0.001 for i in range(len(dataset[0]))])
		self.eta = 0.001
		self.epochs = 1000

	def sigmoid(self, x):
		return 1.0/(1 + math.exp(-1*x))

	def gradientDescent(self, trainingSet):
		for i in range(self.epochs):
			delta = []
			w0 = []
			for data in trainingSet:
				predicted_output = self.sigmoid(sum(np.multiply(self.coeff[1:], data[:4])) + self.coeff[0])

				delta.append(np.multiply(predicted_output - data[-1], data[:4]))
				w0.append(predicted_output - data[-1])

			delta_sum = np.sum(delta, axis=0)
			self.coeff[1:] = self.coeff[1:] - np.multiply(self.eta, delta_sum)
			self.coeff[0] -= self.eta * sum(w0)

		#print self.coeff

	def predict(self, testSet):
		error = []
		for data in testSet:
			predicted_output = self.sigmoid(sum(np.multiply(self.coeff[1:], data[:4])) + self.coeff[0])
			error.append(1 if float(round(predicted_output)) != data[-1] else 0)

		total_error = sum(error)
		#print total_error
		accuracy = (len(testSet) - total_error) / float(len(testSet)) * 100
		#print accuracy
		return accuracy

class PlotData(object):
	def __init__(self):
		plt.xlabel("Data Size (in percentage)")
		plt.ylabel("Accuracy (in percentage)")

	def plotCurve(self, data1, data2, op):
		xAxis = [(i / 5) for i in data2]
		plt.plot(data1, xAxis, op)
	
	def showCurve(self):
		plt.show()

class GenerativeModel(object):
	def __init__(self, dataset):
		self.dataset = dataset

	def splitData(self):
		trainingSize = int(len(self.dataset) * 0.33)
		bin1 = bin2 = bin3 = []
		trainingSizeBin3 = 1 - (2 * int(len(self.dataset) * 0.33))
		while len(bin1) < trainingSize:
			i = random.randrange(len(self.dataset))
			bin1.append(dataset.pop(i))
		
		while len(bin2) < trainingSize:
			i = random.randrange(len(self.dataset))
			bin2.append(dataset.pop(i))

		while len(bin3) < trainingSizeBin3:
			i = random.randrange(len(self.dataset))
			bin3.append(dataset.pop(i))

		return [bin1, bin2, bin3]

	def generateExamples(self):
		bin1, bin2, bin3 = self.splitData()
		dataCombinations = [(bin1+bin2, bin3),(bin2+bin3, bin1),(bin1+bin3, bin2)]
		result = []
		difference = []
		j = 1
		for trainingSet, testSet in dataCombinations:
			obj = GNB()
			labelSummary = obj.summarizeDatasetByLabels(trainingSet) 
			requiredData = labelSummary[1]
			tmpResult = []
			tmpDiff = []
			i = 1
			print "Combination: {}".format(j)
			print "\n"
			for mu, sigma in requiredData:
				samples = np.random.normal(mu, sigma, 400)
				tmpResult.append((obj.mu(samples), obj.sigma(samples)))
				tmpDiff.append((abs(mu - obj.mu(samples)), abs(sigma - obj.sigma(samples))))
				print "Attribute: {}".format(i)
				print "Original Training Set's Mean: {}".format(mu)
				print "Generated Samples' Mean: {}".format(obj.mu(samples))
				print "Difference: {}".format(abs(mu - obj.mu(samples)))
				print "-----------------"
				print "Original Training Set's SD: {}".format(sigma)
				print "Generated Samples' SD: {}".format(obj.sigma(samples))
				print "Difference: {}".format(abs(sigma - obj.sigma(samples)))
				print "-----------------"
				i += 1

			result.append(tmpResult)
			difference.append(tmpDiff)
			print "\n"
			print "***************************"
			print "\n"
			j += 1
		"""print "Generated Samples Data"
		for res in result:
			for features in res:
				print "Mean:", features[0]
				print "Standard Deviation:", features[1]
			print "----------------------------------------"

		print difference"""






if __name__ == "__main__":
	file = 'data_banknote_authentication.csv'
	pd = ProcessData()
	dataset = pd.loadData(file)
	gnb = GNB()
	lr = LogisticRegression(dataset)
	accuracyListGNB = [0,0,0,0,0,0]
	accuracyListLR = [0,0,0,0,0,0]
	for k in range(5):
		data = pd.splitDataSetUsingFractions(dataset)
		j = 0
		for i in range(len(data)):
			trainingSet = data[i][0]
			testSet = data[i][1]
			labelSummaries = gnb.summarizeDatasetByLabels(trainingSet)
			predictions = gnb.getPredictions(labelSummaries, testSet)
			accuracyGNB = gnb.getAccuracy(testSet, predictions)

			lr.gradientDescent(trainingSet)
			accuracyLR = lr.predict(testSet)


			accuracyListGNB[j] += accuracyGNB
			accuracyListLR[j] += accuracyLR
			j += 1
	
	gnbResult = []
	lrResult = []
	for k in range(5):
		gnbResult.append(accuracyListGNB[k]/5)
		lrResult.append(accuracyListLR[k]/5)


	print "Gaussian Naive Bayes Accuracies: ", gnbResult
	print "\n"
	print "Logistic Regression Accuracies:", lrResult
	print "\n"

	fractions = [ .01,  .02,  .05,  .1,  .625,  1]
	plot = PlotData()
	plot.plotCurve(fractions, accuracyListGNB, "bo-")
	plot.plotCurve(fractions, accuracyListLR, "rx-")
	plot.showCurve()

	gm = GenerativeModel(dataset)
	gm.generateExamples()
	



 


