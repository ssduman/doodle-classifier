import numpy as np
import matplotlib.pyplot as plt
from DoodleClassifier import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress cuda warning
import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, layers=None, names=None, l_rate=0.5, epoch=1, from_load=False, tf=False):
        self.layers = layers
        self.L = len(layers) - 1 if layers else 0
        self.names = names
        self.l_rate = l_rate
        self.epoch = epoch
        self.fig, self.ax = plt.subplots()
        self.precision = 0
        self.title = ""

        if from_load:
            self.biases = np.load("biases.npy", allow_pickle=True)
            self.weights = np.load("weights.npy", allow_pickle=True)
            self.L = len(self.names) - 1
        else:
            self.biases = np.array([np.zeros((x, 1)) for x in layers[1:]])
            self.weights = np.array([np.random.randn(layers[x], layers[x - 1]) * np.sqrt(2 / layers[x - 1]) for x in range(1, len(layers))])

        print("layers:", layers)

    def save(self):
        np.save("weights", self.weights, allow_pickle=True)
        np.save("biases", self.biases, allow_pickle=True)
        np.savetxt("names.txt", self.names, delimiter =', ', fmt="%s")

    def feedforward(self, a):
        A = a
        for l in range(self.L - 1):
            output = self.relu(np.dot(self.weights[l], A) + self.biases[l])
            A = output

        output = self.sigmoid(np.dot(self.weights[self.L - 1], A) + self.biases[self.L - 1])

        return output

    def predict(self, test_data):
        test_data = np.array(test_data)
        test_data = test_data.reshape(test_data.shape[0], 1)
        output = self.feedforward(test_data)

        print("output from nn: {} -> sum: {}".format(output, np.sum(output)))
        predictions = np.argmax(output)
        return predictions

    def accuracy(self, test_data, Y):
        test_results = self.feedforward(test_data).T
        predict = 0
        Y = Y.T
        m = test_data.shape[1]
        for i in range(m):
            if np.argmax(test_results[i]) == np.argmax(Y[i]):
                predict += 1
        self.precision = (predict / m) * 100
        self.title += ", acc: %" + str(round(self.precision, 2)) + " (" + str(predict) + "/" + str(m) + ")"
        self.ax.set(xlabel="iterations", ylabel="cost", title=self.title)
        plt.savefig("cost.png")
        # plt.close()   # causes also tkinter to close
        # plt.show()
        return self.precision

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_backward(self, x):
        return (x > 0).astype(float)

    def softmax(self, data):
        exps = [np.exp(x) for x in data]
        sum_of_exps = np.sum(exps)
        soft = np.array([x / sum_of_exps for x in exps])
        return soft

    def softmax_backward(self, x):
        return x * (1. - x)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_backward(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def dropout(self, masks, activation, keep_prob=0.2, layer=0, where="fowardprop"):
        if where == "fowardprop":
            mask = np.random.rand(activation.shape[0], activation.shape[1])
            mask = (mask < keep_prob).astype(int)
            masks.append(mask)
            activation *= mask
            activation /= 0.2
            return activation
        elif where == "backprop":
            activation *= masks[layer]
            activation /= 0.2
            return activation
    
    def L2(self, m, dW=0, lambd=0.2, where="cost"):
        if where == "cost":
            l2 = 0
            for i in range(len(self.weights)):
                l2 += np.sum(np.square(self.weights[i]))
            return (lambd / (2 * m)) * l2
        elif where == "backprop":
            return (lambd / m) * self.weights[dW]
    
    def cost(self, costs, test_costs, m, predict, Y, test_X, test_Y, optimizer="none", _cost="cross_entropy"):
        if _cost == "cross_entropy":
            test_predict = self.feedforward(test_X)
            m_t = float(test_X.shape[1])
            test_cost = (np.sum(test_Y * np.log(test_predict) + (1. - test_Y) * np.log(1. - test_predict))) / -m_t
            test_costs.append(test_cost)

            cost = (np.sum(Y * np.log(predict) + (1. - Y) * np.log(1. - predict))) / -m
            if optimizer == "L2":
                cost += self.L2(m)
            costs.append(cost)

        elif _cost == "mean_square":
            test_predict = self.feedforward(test_X)
            m_t = float(test_X.shape[1])
            test_cost = (np.sum(np.power(test_Y - test_predict, 2))) / m_t
            test_costs.append(test_cost)

            cost = (np.sum(np.power(Y - predict, 2))) / m
            if optimizer == "L2":
                cost += self.L2(m)
            costs.append(cost)

    def train(self, train_X, train_Y, test_X, test_Y, _mini=1024, _cost="cross_entropy", optimizer="none"):
        m = float(train_X.shape[1])
        test_costs = []
        costs = []
        mini = _mini
        print("m: {}, mini: {}, lr: {}".format(m, mini, self.l_rate))
        for x in range(201):
            for y in range(0, int(m), mini):
                m = mini
                data = train_X.T[y:y + mini].T
                Y = train_Y.T[y:y + mini].T
                activation = data
                activations = [activation]
                Zs = []
                masks = []
                grads_W = []
                grads_b = []

                for l in range(self.L - 1):     # fowardprop
                    z = np.dot(self.weights[l], activation) + self.biases[l]
                    Zs.append(z)
                    activation = self.relu(z) 
                    if optimizer == "dropout":
                        activation = self.dropout(masks, activation)
                    activations.append(activation)
                
                z = np.dot(self.weights[self.L - 1], activation) + self.biases[self.L - 1]
                Zs.append(z)
                activation = self.sigmoid(z)
                activations.append(activation)

                dZ = None
                predict = activations[-1]
                if _cost == "cross_entropy":
                    self.cost(costs, test_costs, m, predict, Y, test_X, test_Y, optimizer=optimizer, _cost=_cost)
                    dA = -(np.divide(Y, predict) - np.divide(1. - Y, 1. - predict)) / m
                    dZ = dA * self.sigmoid_backward(Zs[-1])

                if _cost == "mean_square":
                    self.cost(costs, test_costs, m, predict, Y, test_X, test_Y, optimizer=optimizer, _cost=_cost)
                    dA = (Y - predict) / m
                    dZ = dA * self.softmax_backward(Zs[-1])

                dW = np.dot(dZ, activations[-2].T) / m
                if optimizer == "L2":
                    dW += self.L2(m, dW=-1, where="backprop")
                db = np.sum(dZ, axis=1, keepdims=True) / m

                grads_W.append(dW)
                grads_b.append(db)

                for l in reversed(range(self.L - 1)):   # calculate backprop
                    dA = np.dot(self.weights[l + 1].T, dZ)
                    if optimizer == "dropout":
                        dA = self.dropout(masks, dA, layer=l, where="backprop")
                    dZ = dA * self.relu_backward(Zs[l])

                    dW = np.dot(dZ, activations[l].T)
                    if optimizer == "L2":
                        dW += self.L2(m, dW=l, where="backprop")
                    db = np.sum(dZ, axis=1, keepdims=True)

                    grads_W.append(dW)
                    grads_b.append(db)

                for i in range(len(grads_W)):   # update parameters
                    self.weights[-i - 1] -= self.l_rate * grads_W[i]
                    self.biases[-i - 1]  -= self.l_rate * grads_b[i]
            
            if x % 20 == 0:
                print("#: {}, train cost: {:.4f}, test cost: {:.4f}".format(x, costs[-1], test_costs[-1]))
        
        self.title = "#: " + str(Y.shape[0]) + ", l_r: " + str(self.l_rate)
        self.ax.plot(costs, label="train")
        self.ax.plot(test_costs, c="g", label="test")
        self.ax.legend()
        l = "layers: " + self.layers.__str__()
        plt.text(1.02, 0.5, l, rotation=90, ha="left", va="center", transform=self.ax.transAxes)

    def tf(self, x_train, y_train, x_test, y_test):
        x_train = x_train.reshape(x_train.shape[0], 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 28, 28)
        
        temp1 = []
        temp2 = []
        for i in range(y_train.shape[0]):
            temp1.append(np.argmax(y_train[i]))
        for i in range(y_test.shape[0]):
            temp2.append(np.argmax(y_test[i]))
        
        y_train = np.array(temp1)
        y_test = np.array(temp2)

        print("x_train -> X:", x_train.shape, y_train.shape)
        print(" x_test -> X:", x_test.shape, y_test.shape)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.layers[-1], activation="softmax")
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=5, verbose=2)
        acc = model.evaluate(x_test,  y_test)
        print("tf loss: {:.4f}, acc: %{:.4f}".format(acc[0], acc[1] * 100))
