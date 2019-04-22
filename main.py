import neuralnetwork
import itertools
import random

height = 20  # number of hidden layer nodes
width = 1  # width of hidden layer
inputSize = 4  # number of input nodes
outputSize = 1  # number of output nodes
learningRate = 1
epochCount = 999  # number of epoch


def correct(list):
    answer = sum(list) > 1
    return [int(answer)]


squares = []
for i in range(16):
    squares = list(map(list, itertools.product([0, 1], repeat=4)))
print(squares)

NN = neuralnetwork.NeuralNetwork(height, width, inputSize, outputSize)

print(NN.calculate([.25, 0.5, 0.75, 1]))
print(NN.backpropagate([.25, 0.5, 0.75, 1], [1], 0.1))
print("nice")
for i in range(epochCount):
    for square in squares:
        NN.backpropagate(square, correct(square), learningRate)

for square in squares:
    test = random.choice(squares)
    print("Actual:")
    print(correct(test))
    print("NN Calculation")
    print(NN.calculate(test))

