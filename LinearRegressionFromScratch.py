import numpy
inputTrainingData = numpy.array([[1], [2]])
actualValuesofYTraining = numpy.array([[1], [2]])

XTest = numpy.array([[3], [18]])
YTest =  numpy.array([[3], [18]])

class LinearRegressionScratch:
    def __init__(self, XTrain, YTrain):
        self.MAX_ITER = 100000
        self.numberOfInstances, self.numberOfFeatures = XTrain.shape
        self.XTrain = numpy.c_[numpy.ones((len(XTrain), 1)), XTrain]
        self.costCalculatedInEveryIteration = numpy.zeros(self.MAX_ITER)
       # self.inputDataTest = numpy.c_[numpy.ones((len(inputDataFrameTest), 1)), inputDataFrameTest]
       # self.actualValuesOfYTest = actualYTest
        #print(self.inputData)
        self.YTrain = YTrain

        #print("number of features",self.numberOfFeatures)
        self.theta = numpy.random.randn(self.numberOfFeatures+1,1)
       # print(self.theta)
        self.learningRate =0.01


        # print(self.numberOfInstances)
        # print(self.numberOfFeatures)
    def  calculateCostJTheta(self):

        predictions = self.XTrain.dot(self.theta)
        cost = (1/2*self.numberOfInstances) * numpy.sum(numpy.square(predictions - self.YTrain))
        return cost
    def gradientDescent(self):




        for iterationCounter in range(self.MAX_ITER):

            prediction = numpy.dot(self.XTrain, self.theta)

            self.theta = self.theta -((1/self.numberOfInstances) * self.learningRate * (self.XTrain.T.dot((prediction - self.YTrain))))
           # print(self.theta)
            self.costCalculatedInEveryIteration[iterationCounter]  = self.calculateCostJTheta()
           # print(costCalculatedInEveryIteration[iterationCounter])
            if iterationCounter!=0 and self.costCalculatedInEveryIteration[iterationCounter]-self.costCalculatedInEveryIteration[iterationCounter-1]== 0.001:
                print("optimum cost reached")
                break;

        return self.theta,

    def calculateMeanSquaredError(self):
        return self.costCalculatedInEveryIteration.mean()

    def predictorMeanSquareError(self, XTest, YTest):
        XTest= numpy.c_[numpy.ones((len(XTest), 1)), XTest]
        prediction = numpy.dot(XTest, self.theta)
        meanSquareError = (1/self.numberOfInstances) * numpy.sum(numpy.square(prediction - YTest))
        return prediction,meanSquareError
linearRegressionScratchObject = LinearRegressionScratch(inputTrainingData, actualValuesofYTraining)
theta = linearRegressionScratchObject.gradientDescent()

#print(linearRegressionScratchObject.theta)
#print("Mean Squared error for model",linearRegressionScratchObject.calculateMeanSquaredError())


predictedValues,meanSquaredErrorPrediction = linearRegressionScratchObject.predictorMeanSquareError(XTest, YTest)
print("prediction",predictedValues)
print("Mean Squared error for prediction",meanSquaredErrorPrediction)
