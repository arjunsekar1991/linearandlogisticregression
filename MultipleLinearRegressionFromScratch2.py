from builtins import print
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rawData = pandas.read_csv('BSOM_DataSet_for_HW2.csv')
dataWithColumnsRequired = rawData[['all_mcqs_avg_n20','all_NBME_avg_n4','STEP_1']]
from sklearn.linear_model import LinearRegression
#taking only the x values
x = dataWithColumnsRequired.drop('STEP_1',axis=1).values
#print(x)
#taking only y values with mean in the place of na
y = dataWithColumnsRequired.STEP_1.fillna(dataWithColumnsRequired.STEP_1.mean())


XTrain,XTest,YTrain,YTest = train_test_split(x,y,test_size=0.2)
#XTrain = numpy.array([[1], [2]])
#YTrain = numpy.array([[1], [2]])

#XTest = numpy.array([[3], [18]])
#YTest =  numpy.array([[3], [18]])

class LinearRegressionScratch:
    def __init__(self, XTrain, YTrain):
        self.MAX_ITER = 100000
        self.numberOfInstances, self.numberOfFeatures = XTrain.shape
        self.XTrain = numpy.c_[numpy.ones((len(XTrain), 1)), XTrain]
        self.costCalculatedInEveryIteration = numpy.zeros(self.MAX_ITER)
       # self.inputDataTest = numpy.c_[numpy.ones((len(inputDataFrameTest), 1)), inputDataFrameTest]
       # self.actualValuesOfYTest = actualYTest
        #print(self.XTrain)
        self.YTrain = YTrain

        #print("number of features",self.numberOfFeatures)
        #self.theta = numpy.random.randn(self.numberOfFeatures+1,1)
        self.theta = numpy.zeros(self.numberOfFeatures+1)
       # print(self.theta)
        #below is accurate setting
        #self.learningRate =0.01
        self.learningRate =0.01


        # print(self.numberOfInstances)
        # print(self.numberOfFeatures)
    def costPlot(self):
        fig, ax = plt.subplots(figsize=(8,8))
        print("values of jtheta",self.costCalculatedInEveryIteration)
        plt.plot(self.costCalculatedInEveryIteration)
        plt.ylabel('Epochs')
        plt.xlabel('Cost jThetha')
        plt.show()
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
            if iterationCounter!=0 and self.costCalculatedInEveryIteration[iterationCounter]>self.costCalculatedInEveryIteration[iterationCounter-1]:
                print("optimum cost reached")
                break;

        return self.theta,

    def calculateMeanSquaredError(self):
        return self.costCalculatedInEveryIteration.mean()

    def predictorMeanSquareError(self, XTest, YTest):
        XTest= numpy.c_[numpy.ones((len(XTest), 1)), XTest]
        prediction = numpy.dot(XTest, self.theta)
        numberOfTestInstance = len(YTest)
        meanSquareError = (1/numberOfTestInstance) * numpy.sum(numpy.square(prediction - YTest))
        return prediction,meanSquareError
    def showModel(self,XTest,YTest):
        XTest = numpy.sum(XTest,axis=1)
        print("xtest1",XTest)
        print("xtest",min(numpy.array(XTest)))
        print("ytest",min(numpy.array(YTest)))
        plt.xlabel("XTest")
        plt.ylabel("YTest")
        plt.scatter(XTest,YTest)

        plt.plot([min(numpy.array(XTest)),max(numpy.array(XTest))],[min(numpy.array(YTest)),max(numpy.array(YTest))],color ="yellow")
        plt.show()
linearRegressionScratchObject = LinearRegressionScratch(XTrain, YTrain)
theta = linearRegressionScratchObject.gradientDescent()
linearRegressionScratchObject.costPlot()
print(linearRegressionScratchObject.theta)
#print("Mean Squared error for model",linearRegressionScratchObject.calculateMeanSquaredError())


predictedValues,meanSquaredErrorPrediction = linearRegressionScratchObject.predictorMeanSquareError(XTest, YTest)
print("prediction",predictedValues)
print("Mean Squared error for prediction",meanSquaredErrorPrediction)
linearRegressionScratchObject.showModel(XTest, YTest)
print(mean_squared_error(YTest, predictedValues, multioutput='raw_values'))
print(r2_score(YTest, predictedValues))

reg = LinearRegression()
model = reg.fit(XTrain, YTrain)
print(reg.score(XTest,YTest))
