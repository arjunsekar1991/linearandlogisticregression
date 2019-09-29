import numpy
inputTrainingData = numpy.array([[1,1], [2,2]])
actualValuesofYTraining = numpy.array([[1,1], [2,2]])

inputTestData = numpy.array([[3,3],[4,4]])
actualValuesofYTest =  numpy.array([[3,3],[4,4]])

class LinearRegressionScratch:
    def __init__(self, inputDataFrameTraining, actualYTraining):
        self.MAX_ITER = 100000
        self.numberOfInstances, self.numberOfFeatures = inputDataFrameTraining.shape
        self.inputDataTraining = numpy.c_[numpy.ones((len(inputDataFrameTraining), 1)), inputDataFrameTraining]
        self.costCalculatedInEveryIteration = numpy.zeros(self.MAX_ITER)
       # self.inputDataTest = numpy.c_[numpy.ones((len(inputDataFrameTest), 1)), inputDataFrameTest]
       # self.actualValuesOfYTest = actualYTest
        #print(self.inputData)
        self.actualValuesOfY = actualYTraining

        #print("number of features",self.numberOfFeatures)
        self.theta = numpy.random.randn(self.numberOfFeatures+1,1)
       # print(self.theta)
        self.learningRate =0.01


        # print(self.numberOfInstances)
        # print(self.numberOfFeatures)
    def  calculateCostJTheta(self):

        predictions = self.inputDataTraining.dot(self.theta)
        cost = (1/2*self.numberOfInstances) * numpy.sum(numpy.square(predictions-self.actualValuesOfY))
        return cost
    def gradientDescent(self):




        for iterationCounter in range(self.MAX_ITER):

            prediction = numpy.dot(self.inputDataTraining, self.theta)

            self.theta = self.theta -((1/self.numberOfInstances) * self.learningRate * (self.inputDataTraining.T.dot((prediction - self.actualValuesOfY))))
           # print(self.theta)
            self.costCalculatedInEveryIteration[iterationCounter]  = self.calculateCostJTheta()
           # print(costCalculatedInEveryIteration[iterationCounter])
            if iterationCounter!=0 and self.costCalculatedInEveryIteration[iterationCounter]-self.costCalculatedInEveryIteration[iterationCounter-1]== 0.001:
                print("optimum cost reached")
                break;

        return self.theta,

    def calculateMeanSquaredError(self):
        return self.costCalculatedInEveryIteration.mean()

    def predictor(self,inputTestData):
        inputTestData= numpy.c_[numpy.ones((len(inputTestData), 1)), inputTestData]
        prediction = numpy.dot(inputTestData, self.theta)
        return prediction
linearRegressionScratchObject = LinearRegressionScratch(inputTrainingData, actualValuesofYTraining)
theta = linearRegressionScratchObject.gradientDescent()

print(linearRegressionScratchObject.theta)
print("Mean Squared error",linearRegressionScratchObject.calculateMeanSquaredError())


predictedValues = linearRegressionScratchObject.predictor(inputTestData)
print("prediction",predictedValues)
