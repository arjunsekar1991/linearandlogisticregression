import numpy
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pandas
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


class LogisticRegression:

    def __init__(self, XTrain, YTrain, normalization=True):
        self.MAX_ITER = 1000000
        #print(XTrain)x`
        self.numberOfInstances, self.numberOfFeatures = XTrain.shape
       # print(self.numberOfFeatures)
        self.XTrain = XTrain
        self.numberOfClasses = len(numpy.unique(YTrain))
        #print(self.numberOfClasses)
        self.YTrain = YTrain.to_numpy().reshape(self.numberOfInstances,1)
        self.theta = []
        self.costCalculated = []
      #  print(self.theta)
        self.learningRate =0.7
      #  print(self.numberOfClasses)
    def  calculateCostJTheta(self,theta,YTrainOneVSALL):
       # print("theta",theta)
        predictions =  1/(1+numpy.exp(-self.XTrain.dot(theta)))
        #print("predictions",predictions)
        #for x in predictions:
        error = (-YTrainOneVSALL * numpy.log(predictions)) - ((1-YTrainOneVSALL)*numpy.log(1-predictions))
        #print("error",error)
        cost = 1/self.numberOfInstances * sum(error)
        #print("every",cost,"\n")
        return cost

    def gradientDescent(self):
        nonregulazied =self.XTrain
        #print(nonregulazied)
        regularized = (self.XTrain - self.XTrain.mean())/(self.XTrain.max()-self.XTrain.min())
        #regularized= self.featureScalingUsingMinMaxNormalization(nonregulazied)
        #print(regularized)
        self.XTrain = numpy.c_[numpy.ones((len(nonregulazied), 1)), nonregulazied]


        #th =  numpy.random.rand(self.numberOfFeatures+1,1)
        th =  numpy.zeros(self.numberOfFeatures+1).reshape(self.numberOfFeatures+1,1)
        print(th)
       # print(self.XTrain)

        for classCounter in range(self.numberOfClasses):
            costCalculatedInEveryIterationForEachClass = numpy.zeros(self.MAX_ITER)
            YTrainOneVSALL = numpy.where(self.YTrain == classCounter, 1, 0)
           # print(YTrainOneVSALL)
            i = 0
            print("class",classCounter)
            while (i < self.MAX_ITER):

                costCalculatedInEveryIterationForEachClass[i] = self.calculateCostJTheta(th,YTrainOneVSALL)
                #if i==1:
                #    print("inital cost",costCalculatedInEveryIterationForEachClass[i])
                #if i==200:
                #    print("cost at 100",costCalculatedInEveryIterationForEachClass[i])
                #if i==400:
                #    print("cost at 200",costCalculatedInEveryIterationForEachClass[i])
                #if i==600:
                #    print("cost at 600",costCalculatedInEveryIterationForEachClass[i])
                if i==800:
                    print("cost at 800",costCalculatedInEveryIterationForEachClass[i])
                if i!=0:
                    #thres =costCalculatedInEveryIterationForEachClass[i]-costCalculatedInEveryIterationForEachClass[i-1]
                    #print(costCalculatedInEveryIterationForEachClass[i])
                    if costCalculatedInEveryIterationForEachClass[i]>costCalculatedInEveryIterationForEachClass[i-1]:
                        #print(thres)
                        print("optimum cost reached at" , i)
                        break
                prediction = 1/(1+numpy.exp(-self.XTrain.dot(th)))
                th = th - (self.learningRate / self.numberOfInstances) * (self.XTrain.T).dot(prediction - YTrainOneVSALL)

                i+=1
            self.theta.append(th)
            self.costCalculated.append(costCalculatedInEveryIterationForEachClass)


        return self.theta

    def predict(self,X):
               # X = self.featureScalingUsingMinMaxNormalization(X)
                #X=(X - X.mean())/(X.max()-X.min())
                X = numpy.c_[numpy.ones((len(X), 1)), X]
                #print(X)
                maximumLikelihood = []
                #data=pandas.DataFrame(X)
                finalPrediction = []
                for i in range(self.numberOfClasses):

                    thv = self.theta[i]
                    predictions =  1/(1+numpy.exp(-X.dot(thv)))
                   # pandas.concat(data,predictions)
                    #print("predictions",predictions)
                    maximumLikelihood.append(numpy.array(predictions).flatten())
                    #maximumLikelihood.append(list(zip(*predictions)))
                    #maximumLikelihood.append(numpy.array(predictions))
                #print("final",numpy.vstack( maximumLikelihood ))
                ##print(data)
              # for k in len(X):
                #print(maximumLikelihood)
                data_tuples = list(zip(maximumLikelihood[0],maximumLikelihood[1],maximumLikelihood[2],maximumLikelihood[3]))

                #print(data_tuples)
                likelihoodlist =[]
                for tuplecounter in range(len(data_tuples)):
                    likelihoodlist.append(numpy.array(data_tuples[tuplecounter]))
                #print("final",likelihoodlist)
                for k in range(len(likelihoodlist)):
                    finalPrediction.append(numpy.argmax(likelihoodlist[k]))
                #print(numpy.array(maximumLikelihood[0]))
                #print("first",maximumLikelihood[0][0])
                #print(numpy.array(maximumLikelihood[1]))
                #print(numpy.array(maximumLikelihood[2]))
                #print(numpy.array(maximumLikelihood[3]))

                #print(maximumLikelihood[1])
                #print(maximumLikelihood[2])
                #print(maximumLikelihood[3])
               # maximumLikelihood = [max(x) for x in list(zip(*maximumLikelihood))]

                return finalPrediction

rawData = pandas.read_csv('BSOM_DataSet_for_HW2.csv')
dataWithColumnsRequired = rawData[['all_NBME_avg_n4','all_mcqs_avg_n20','LEVEL']]
dataWithColumnsRequiredWithoutNull = dataWithColumnsRequired.dropna(axis = 0, how ='any')


x = dataWithColumnsRequiredWithoutNull.drop('LEVEL',axis=1).values
ynonfactor = dataWithColumnsRequiredWithoutNull.LEVEL

y= ynonfactor.replace(to_replace=['A', 'B','C','D'], value=[0,1,2,3])

#print(y)
XTrain,XTest,YTrain,YTest = train_test_split(x,y,test_size=0.2,random_state=0)
print()


logisticRegressionObject = LogisticRegression(XTrain, YTrain)
#from sklearn.cross_validation
from sklearn import metrics
#scores = cross_val_score(logisticRegressionObject, XTrain, YTrain, cv=6)
theta = logisticRegressionObject.gradientDescent()
print(theta)

pv = logisticRegressionObject.predict(XTest)
#print(numpy.c_[y_test,pv])x
print(pv)
a = numpy.array(YTest,dtype=int).flatten().tolist()

print(a)
print(confusion_matrix(a, pv,labels=numpy.unique(YTrain)))

sn.heatmap(confusion_matrix(a,pv),annot=True,fmt='g',yticklabels=numpy.unique(YTrain),xticklabels=numpy.unique(YTrain));
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
#con.plot()
print(f1_score(a, pv, average='macro'))

#clf = LogisticRegression()


#clf.fit(XTrain,YTrain)
#print(clf.theta)
