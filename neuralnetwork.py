import math
import random
# [0.5] * (input if self.neurons == [] else self.neurons[i-1].size()))


class NeuralNetwork:
    def __init__(self, height, width, inputs, output):
        self.neurons = []
        layer = []

        for i in range(width):  # adding middle layer neurons
            for j in range(height):
                layer.append(Neuron([]))
            self.neurons.append(layer)
            layer = []

        for i in range(output):  # adding output layer neurons
            layer.append(Neuron([]))
        self.neurons.append(layer)

        for n in self.neurons[0]:  # adding first layer of weights
            for i in range(inputs):
                n.weights.append(random.random())

        for l in self.neurons[1:]:  # adding middle weight layers
            for n in l:  # for neuron in layer
                for i in range(height):
                    n.weights.append(random.random())

    def calculate(self, inputs):
        for n in self.neurons[0]:
            n.calculate(inputs)
        for i, l in enumerate(self.neurons[1:], start=1):
            for n in l:  # for neuron in layer
                n.calculate([o.result for o in self.neurons[i-1]])
        return [o.result for o in self.neurons[-1]]

    def backpropagate(self, inputs, target, learning_rate):
        self.calculate(inputs)

        # calculating error
        error = []
        layer = []
        for i, n in enumerate(self.neurons[-1]):
            layer.append(n.result * (1 - n.result) * (target[i] - n.result))
        error.append(layer)
        layer = []
        for i, l in reversed(list(enumerate(self.neurons[:-1]))):
            for j, n in enumerate(l):
                layer.append(n.result * (1 - n.result) * Neuron.dot([o.weights[j] for o in self.neurons[i + 1]], error[-1]))
            error.append(layer)
            layer = []
        error.reverse()
        # print(error)

        # reweighting
        for j, n in enumerate(self.neurons[0]):
            for k, weight in enumerate(n.weights):
                n.weights[k] = weight + (learning_rate * error[0][j] * inputs[k])

        for i, l in enumerate(self.neurons[1:]):
            for j, n in enumerate(l):
                for k, weight in enumerate(n.weights):
                    n.weights[k] = weight + learning_rate * error[i + 1][j] * self.neurons[i][k].result



class Neuron:
    def __init__(self, weights):
        self.result = -99
        self.weights = [float(i) for i in weights]

    def calculate(self, output):
        input = self.dot(output, self.weights)

        self.result = 1 / (1 + math.exp(-input))
        return self.result

    def get_result(self):
        return self.result

    @staticmethod
    def dot(a, b):
        if len(a) != len(b):
            return 0
        return sum(i[0] * i[1] for i in zip(a, b))
