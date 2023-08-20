import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for our network

inputNodes = 2
outputNodes = 1
hiddenNodes = 3
batchSize = 10 # how much data we want to put through the network at one time

with open("data.txt", "r") as f:
    data = f.readlines()

for idx, line in enumerate(data):
    parts = line.split(' ')
    parts[2] = parts[2].replace('\n', '')  # Corrected line to replace newline
    parts[0] = int.from_bytes(parts[0].encode(), "big")
    parts[1] = int(parts[1], base=16)
    parts[0] = "".join(map(str, parts[0:2]))
    del parts[1]
    data[idx] = " ".join(map(str, parts))  # Join the modified parts back to a line

for idx, line in enumerate(data):
    line = line.split(" ")
    newData = []
    for i in line[0]:
        newData.append(int(i))
    newData.append(int(line[1]))
    data[idx] = newData
inputData = []
outputData = []
for i in data:
    outputData.append(i[-1])
    newList = []
    for j in i[:-1]:
        newList.append(j / 10)
    inputData.append(newList)

for i in inputData:
    while len(i) < 105:
        i.insert(0, 0)

# inputData = [numpy.array(batch) for batch in inputData]
# outputData = numpy.array(outputData)

trainingInputData = inputData[0:50]
trainingInputData = numpy.array(trainingInputData)
testingInputData = inputData[50:]
testingInputData = numpy.array(testingInputData)

trainingOutputData = outputData[0:50]
trainingOutputData = numpy.array(trainingOutputData)
testingOutputData = outputData[50:]
testingOutputData = numpy.array(testingOutputData)

# inputData = numpy.random.randn(batchSize, inputNodes)
# outputData = numpy.random.randn(batchSize, outputNodes)

weightsMatrix1 = numpy.random.randn(len(inputData[0]), hiddenNodes)
weightsMatrix2 = numpy.random.randn(hiddenNodes, 1)
with open('weightsMatrix1.txt', 'r') as f:
    pass
with open('weightsMatrix2.txt', 'r') as f:
    pass

lossArray = numpy.array([[]])
indices = numpy.array([[]])

def train(inputData, outputData):
    global weightsMatrix1, weightsMatrix2, lossArray, indices
    for i in range(1000):
        for inputBatch, outputBatch in zip(inputData, outputData):
            hiddenValues = inputBatch.dot(weightsMatrix1)

            # Rectified Linear Unit (ReLU)
            # renives akk begatuve vakyes, replacing with 0

            hiddenRelu = numpy.maximum(hiddenValues, 0)
            outputDataPredictions = hiddenRelu.dot(weightsMatrix2)
            lossFunction = numpy.square(outputDataPredictions - outputBatch).sum()
            lossArray = numpy.append(lossArray, lossFunction)
            indices = numpy.append(indices, i)

            #Back propogation
            gradientPrediction = 2 * (outputDataPredictions - outputBatch)
            gradientWeights2 =  hiddenRelu.reshape((-1, 1)).dot(gradientPrediction.reshape((1, -1)))
            gradientHiddenRelu = gradientPrediction.dot(weightsMatrix2.T)
            gradientHiddenValues = gradientHiddenRelu.copy()
            gradientHiddenValues[hiddenValues < 0] = 0
            gradientWeights1 = inputBatch.T.reshape((-1,1)).dot(gradientHiddenValues.reshape((1, -1)))
            weightsMatrix1 = weightsMatrix1 - gradientWeights1 * 1e-4
            weightsMatrix2 = weightsMatrix2 - gradientWeights2 * 1e-4
        with open('weightsMatrix1.txt', 'w') as f:
            f.write(str(weightsMatrix1))
        with open('weightsMatrix2.txt', 'w') as f:
            f.write(str(weightsMatrix2))

train(trainingInputData, trainingOutputData)

plt.plot(indices, lossArray)
plt.legend(['Loss over iterations'])
plt.show()
