import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for our network

inputNodes = 2
outputNodes = 2
hiddenNodes = 3
batchSize = 8 # how much data we want to put through the network at one time

inputData = numpy.random.randn(batchSize, inputNodes)
outputData = numpy.random.randn(batchSize, outputNodes)

weightsMatrix1 = numpy.random.randn(inputNodes, hiddenNodes)
weightsMatrix2 = numpy.random.randn(hiddenNodes, outputNodes)

lossArray = numpy.array([[]])
indices = numpy.array([[]])

for i in range(10000):
    hiddenValues = inputData.dot(weightsMatrix1)

    # Rectified Linear Unit (ReLU)
    # renives akk begatuve vakyes, replacing with 0

    hiddenRelu = numpy.maximum(hiddenValues, 0)
    outputDataPredictions = hiddenRelu.dot(weightsMatrix2)
    lossFunction = numpy.square(outputDataPredictions - outputData).sum()
    lossArray = numpy.append(lossArray, lossFunction)
    indices = numpy.append(indices, i)

    #Back propogation
    gradientPrediction = 2 * (outputDataPredictions - outputData)
    gradientWeights2 =  hiddenRelu.T.dot(gradientPrediction)
    gradientHiddenRelu = gradientPrediction.dot(weightsMatrix2.T)
    gradientHiddenValues = gradientHiddenRelu.copy()
    gradientHiddenValues[hiddenValues < 0] = 0
    gradientWeights1 = inputData.T.dot(gradientHiddenValues)
    weightsMatrix1 = weightsMatrix1 - gradientWeights1 * 1e-4


plt.plot(indices, lossArray)
plt.legend(['Loss over iterations'])
plt.show()

print(outputData)
print(outputDataPredictions)