import numpy
inputTestData = numpy.array([[1],[2]])
actualValuesofY = numpy.array([[1],[2]])
class LinearRegressionScratch:
    def __init__(self, inputDataFrame,actualY):
        self.inputData = inputDataFrame
        self.actualValuesOfY = actualY
        self.numberOfInstances, self.numberOfFeatures = self.inputData.shape
        self.learningRate =0.01
        self.MAX_ITER = 1000

        # print(self.numberOfInstances)
        # print(self.numberOfFeatures)
    def  calculateCostJTheta(self,theta):

        predictions = self.inputData.dot(theta)
        cost = (1/2*self.numberOfInstances) * numpy.sum(numpy.square(predictions-self.actualValuesOfY))
        return cost
