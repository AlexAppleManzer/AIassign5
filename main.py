import neuralnetwork
import itertools
import random

height = 10  # number of hidden layer nodes
width = 1  # width of hidden layer
inputSize = 4  # number of input nodes
outputSize = 1  # number of output nodes
learningRate = 10
epochCount = 999  # number of epoch


def correct(list):
    answer = sum(list) > 1
    return [int(answer)]


def squarestest():
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


#squarestest()


def rock_paper_scissors():

    def convert(char):
        if char == 'R':
            return 0
        elif char == 'P':
            return 1
        elif char == 'S':
            return 2

    def go_back(i):
        if i == 0:
            return 'P'
        elif i == 1:
            return 'S'
        elif i == 2:
            return 'R'

    def to_input(o, t):
        output = []
        if o == 0:
            output.append(1)
            output.append(0)
            output.append(0)
        elif o == 1:
            output.append(0)
            output.append(1)
            output.append(0)
        elif o == 2:
            output.append(0)
            output.append(0)
            output.append(1)
        if t == 0:
            output.append(1)
            output.append(0)
            output.append(0)
        elif t == 1:
            output.append(0)
            output.append(1)
            output.append(0)
        elif t == 2:
            output.append(0)
            output.append(0)
            output.append(1)
        return output

    def to_output(c):
        output = []
        if c == 0:
            output.append(1)
            output.append(0)
            output.append(0)
        elif c == 1:
            output.append(0)
            output.append(1)
            output.append(0)
        elif c == 2:
            output.append(0)
            output.append(0)
            output.append(1)
        return output

    def choose(l):
        result = -1
        weight = -1
        for i, item in enumerate(l):
            if item > weight:
                result = i
                weight = item
        return result


    NN = neuralnetwork.NeuralNetwork(height, width, 6, 3)

    print("Rock.. Paper.. Scissors... (R P S)")
    one = convert(input())
    print(go_back(random.randint(0, 2)))

    print("Rock.. Paper.. Scissors... (R P S)")
    two = convert(input())
    print(go_back(random.randint(0, 2)))

    while True:
        print("Rock.. Paper.. Scissors... (R P S)")
        out = convert(input())
        guess = NN.calculate(to_input(one, two))
        print("Guess:")
        print(guess)
        print("Real:")
        print(to_output(out))
        print(go_back(choose(guess)))
        NN.backpropagate(to_input(one, two), to_output(out), 1)
        one = two
        two = out

rock_paper_scissors()
