import numpy as np
from DoodleClassifier import *

class NeuralNetwork(object):
    def __init__(self, layers=None, names=None, l_rate=0.1, epoch=1, from_load=False):
        self.layers = layers
        self.layer_number = len(layers) if layers else 0
        self.names = names
        self.l_rate = l_rate
        self.epoch = epoch

        if from_load:
            self.biases = np.load("biases.npy", allow_pickle=True)
            self.weights = np.load("weights.npy", allow_pickle=True)
            self.layer_number = len(self.names)
        else:
            self.biases = np.array([np.random.randn(y, ) for y in layers[1:]])
            self.weights = np.array([np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])])

    def save(self):
        np.save("weights", self.weights, allow_pickle=True)
        np.save("biases", self.biases, allow_pickle=True)
        np.savetxt("names.txt", self.names, delimiter=', ', fmt="%s")

    def feedforward(self, a):
        input = a
        output = None
        for b, w in zip(self.biases, self.weights):
            output = self.sigmoid(np.dot(w, input) + b)
            input = output

        return output

    def predict(self, test_data):
        output = self.feedforward(test_data)

        exps = [np.exp(x) for x in output]
        sum_of_exps = np.sum(exps)
        softmax = [x / sum_of_exps for x in exps]

        print("output from nn: {}\n, softmax: {}\n, sum: {}\n".format(output, softmax, np.sum(softmax)))
        predictions = np.argmax(output)
        return predictions

    def accuracy(self, test_data):
        test_results = np.array([self.feedforward(x) for (x, y) in test_data])
        predict = 0
        for test in test_results:
            if np.argmax(test) == np.argmax(test_data[0][1]):
                predict += 1
        return (predict / len(test_data)) * 100

    def train(self, data, single=False):
        for j in range(self.epoch):
            if not single:
                for x, y in data:
                    self.back_propagation(x, y)
            else:
                self.back_propagation(data[0], data[1])

    def back_propagation(self, x, y):
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        error = y - activations[-1]
        delta = error * self.sigmoid_prime(zs[-1])
        self.biases[-1] = np.add(self.biases[-1], delta)
        self.weights[-1] = np.add(self.weights[-1], np.outer(delta, activations[-2].transpose()))

        for i in range(2, self.layer_number):
            z = zs[-i]
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * self.sigmoid_prime(z)
            self.biases[-i] = np.add(self.biases[-i], delta)
            self.weights[-i] = np.add(self.weights[-i], np.outer(delta, activations[-i - 1].transpose()))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

