import numpy, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for our network

inputNodes = 2
outputNodes = 1
hiddenNodes = 1000
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
    newData.append(float(line[1]))
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

trainingInputData = inputData
trainingInputData = numpy.array(trainingInputData)
testingInputData = inputData
testingInputData = numpy.array(testingInputData)

trainingOutputData = outputData
trainingOutputData = numpy.array(trainingOutputData)
testingOutputData = outputData
testingOutputData = numpy.array(testingOutputData)

# weightsMatrix1 = numpy.random.randn(len(inputData[0]), hiddenNodes)
# weightsMatrix2 = numpy.random.randn(hiddenNodes, 1)

weightsMatrix1 = numpy.loadtxt("weightsMatrix1.txt")
weightsMatrix1 = numpy.array(weightsMatrix1).reshape((105, hiddenNodes))

weightsMatrix2 = numpy.loadtxt("weightsMatrix2.txt")
weightsMatrix2 = numpy.array(weightsMatrix2).reshape((hiddenNodes,1))


lossArray = numpy.array([[]])
indices = numpy.array([[]])

def train(inputData, outputData):
    global weightsMatrix1, weightsMatrix2, lossArray, indices
    for i in range(100):
        for inputBatch, outputBatch in zip(inputData, outputData):
            hiddenValues = inputBatch.dot(weightsMatrix1)

            # Rectified Linear Unit (ReLU)
            # renives akk begatuve vakyes, replacing with 0

            hiddenRelu = numpy.maximum(hiddenValues, 0)
            outputDataPredictions = hiddenRelu.dot(weightsMatrix2)
            lossFunction = numpy.square(outputDataPredictions - outputBatch).sum()

            #Back propogation
            gradientPrediction = 2 * (outputDataPredictions - outputBatch)
            gradientWeights2 =  hiddenRelu.reshape((-1, 1)).dot(gradientPrediction.reshape((1, -1)))
            gradientHiddenRelu = gradientPrediction.dot(weightsMatrix2.T)
            gradientHiddenValues = gradientHiddenRelu.copy()
            gradientHiddenValues[hiddenValues < 0] = 0
            gradientWeights1 = inputBatch.T.reshape((-1,1)).dot(gradientHiddenValues.reshape((1, -1)))
            weightsMatrix1 = weightsMatrix1 - gradientWeights1 * 1e-4
            weightsMatrix2 = weightsMatrix2 - gradientWeights2 * 1e-4
        #shuffling training data
        indices2 = list(range(len(inputData)))
        random.shuffle(indices2)

        inputData = [inputData[i] for i in indices2]
        outputData = [outputData[i] for i in indices2]

        numpy.savetxt("weightsMatrix1.txt", weightsMatrix1, fmt="%e")
        numpy.savetxt("weightsMatrix2.txt", weightsMatrix2, fmt="%e")

        lossArray = numpy.append(lossArray, lossFunction)
        indices = numpy.append(indices, i)

def testModel(inputData, outputData):
    global weightsMatrix1, weightsMatrix2, lossArray, indices
    count = 0
    for inputBatch, outputBatch in zip(inputData, outputData):
        hiddenValues = inputBatch.dot(weightsMatrix1)

        # Rectified Linear Unit (ReLU)
        # renives akk begatuve vakyes, replacing with 0

        hiddenRelu = numpy.maximum(hiddenValues, 0)
        outputDataPredictions = hiddenRelu.dot(weightsMatrix2)
        lossFunction = numpy.square(outputDataPredictions - outputBatch).sum()
        lossArray = numpy.append(lossArray, lossFunction)
        indices = numpy.append(indices, count)
        count = count + 1

def predict():
    global weightsMatrix1, weightsMatrix2, lossArray, indices
    count = 0
    for i in range(10):
        inputData = []
        inputSeed = input("Seed: ")
        firstPart = int.from_bytes(inputSeed.encode(), "big")
        inputServerSeed = input("Server Seed: ")
        secondPart = int(inputServerSeed, base=16)
        for i in str(firstPart):
            inputData.append(int(i)/10)
        for i in str(secondPart):
            inputData.append(int(i)/10)
        while len(inputData) < 105:
            inputData.insert(0, 0)
        inputData = numpy.array(inputData)

        hiddenValues = inputData.dot(weightsMatrix1)

        # Rectified Linear Unit (ReLU)
        # renives akk begatuve vakyes, replacing with 0

        hiddenRelu = numpy.maximum(hiddenValues, 0)
        outputDataPredictions = hiddenRelu.dot(weightsMatrix2)
        print(outputDataPredictions)
        correctAnswer = float(input("Enter the correct answer: "))
        lossFunction = numpy.square(outputDataPredictions - correctAnswer).sum()
        lossArray = numpy.append(lossArray, lossFunction)
        indices = numpy.append(indices, count)

        #Back propogation
        gradientPrediction = 2 * (outputDataPredictions - correctAnswer)
        gradientWeights2 =  hiddenRelu.reshape((-1, 1)).dot(gradientPrediction.reshape((1, -1)))
        gradientHiddenRelu = gradientPrediction.dot(weightsMatrix2.T)
        gradientHiddenValues = gradientHiddenRelu.copy()
        gradientHiddenValues[hiddenValues < 0] = 0
        gradientWeights1 = inputData.T.reshape((-1,1)).dot(gradientHiddenValues.reshape((1, -1)))
        weightsMatrix1 = weightsMatrix1 - gradientWeights1 * 1e-4
        weightsMatrix2 = weightsMatrix2 - gradientWeights2 * 1e-4
        numpy.savetxt("weightsMatrix1.txt", weightsMatrix1, fmt="%e")
        numpy.savetxt("weightsMatrix2.txt", weightsMatrix2, fmt="%e")
        count = count + 1

#train(trainingInputData, trainingOutputData)
    
#testModel(testingInputData, testingOutputData)

predict()

plt.plot(indices, lossArray)
plt.legend(['Loss over iterations'])
plt.show()