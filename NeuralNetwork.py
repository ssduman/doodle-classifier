import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress cuda warning
# import tensorflow as tf
from DoodleClassifier import *

class NeuralNetwork(object):
    def __init__(self, layers=None, names=None, from_load=False, tf=False):
        self.layers = layers
        self.L = len(layers) - 1 if layers else 0
        self.names = names
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8,
        self.soft = False
        self.reset_count = 0
        self.config = {
            "l_rate" : 0.01, 
            "epoch" : 5, 
            "batch_size" : 256, 
            "loss" : "cross_entropy",   # "cross_entropy", "multi_label", "mean_square"
            "optimization" : "adam",    # "adam", "momentum"
            "regularization" : "none"   # "dropout", "L2"
        }

        if from_load:
            self.biases = np.load("biases.npy", allow_pickle=True)
            self.weights = np.load("weights.npy", allow_pickle=True)
            self.L = 2
        else:
            self.biases = np.array([np.zeros((x, 1)) for x in layers[1:]])
            self.weights = np.array([np.random.randn(layers[x], layers[x - 1]) * np.sqrt(2 / layers[x - 1]) for x in range(1, len(layers))])

        print("layers: {}, len: {}".format(self.layers, self.L))

    def feedforward(self, a):
        A = a
        for l in range(self.L - 1):
            output = self.relu(np.dot(self.weights[l], A) + self.biases[l])
            A = output

        if self.soft:
            z = np.dot(self.weights[self.L - 1], A) + self.biases[self.L - 1]
            output = self.softmax(z - np.max(z))    # self.softmax(z)
            # output = self.softmax(z)    # self.softmax(z)
        else:
            output = self.sigmoid(np.dot(self.weights[self.L - 1], A) + self.biases[self.L - 1])

        return output

    def train(self, train_X, train_Y, test_X, test_Y, config={}):
        assert(train_X.shape[0] == self.weights[0].shape[1])  # train_X.shape -> (input_size, m)
        assert(train_Y.shape[0] == self.layers[-1])

        self.config = {**self.config, **config}
        l_rate = self.config["l_rate"]
        epoch = self.config["epoch"]
        batch_size = self.config["batch_size"]
        loss = self.config["loss"]
        optimization  = self.config["optimization"]
        regularization = self.config["regularization"]

        for key, value in self.config.items():
            print("\"{}\" : {}".format(key, value))

        m = float(train_X.shape[1])
        print("m: {}".format(m))
        test_costs = []
        costs = []
        test_accs = []
        accs = []
        moments = [[]]
        adams = [[]]

        if optimization == "momentum":
            for l in range(self.L):
                moments.append([np.zeros(self.weights[l].shape), np.zeros(self.biases[l].shape)])
        if optimization == "adam":
            for l in range(self.L):
                moments.append([np.zeros(self.weights[l].shape), np.zeros(self.biases[l].shape)])
                adams.append([np.zeros(self.weights[l].shape), np.zeros(self.biases[l].shape)])
        
        if loss == "multi_label":
            print("sotmax activated")
            self.soft = True
        
        threshold = 0.5

        self.cost(costs, test_costs, m, train_X, train_Y, test_X, test_Y)
        accs.append(self.accuracy(train_X, train_Y))
        test_accs.append(self.accuracy(test_X, test_Y))
        # print("first acc:", accs, test_accs)
        print("{}/{}, train cost: {:.4f}, test cost: {:.4f}".format(0, epoch, costs[-1], test_costs[-1]))
        for x in range(epoch):
            for y in range(0, int(m), batch_size):
                if y + batch_size >= m: y = int(m - batch_size - 1)
                data = train_X[:, y:y + batch_size]
                Y = train_Y[:, y:y + batch_size]
                activation = data
                activations = [activation]
                Zs = []
                masks = []
                grads_W = []
                grads_b = []

                for l in range(self.L - 1):  # fowardprop
                    z = np.dot(self.weights[l], activation) + self.biases[l]
                    Zs.append(z)
                    activation = self.relu(z)
                    if regularization == "dropout":
                        activation = self.dropout(masks, activation)
                    activations.append(activation)

                z = np.dot(self.weights[self.L - 1], activation) + self.biases[self.L - 1]
                Zs.append(z)
                if self.soft:
                    activation = self.softmax(z - np.max(z))
                else:
                    activation = self.sigmoid(z)
                activations.append(activation)
                
                dZ = None
                predict = activations[-1]
                if loss == "cross_entropy":
                    dA = -(np.divide(Y, predict) - np.divide(1. - Y, 1. - predict)) / batch_size  # dCost / dPredict
                    dZ = dA * self.sigmoid_backward(Zs[-1]) # dPredict / dZ[last]

                if loss == "multi_label":
                    dZ = (predict - Y) / batch_size

                if loss == "mean_square":
                    dA = (predict - Y) / batch_size
                    dZ = dA * self.sigmoid_backward(Zs[-1])

                dW = np.dot(dZ, activations[-2].T) / batch_size   # dZ[last] / dW[last]
                if regularization == "L2":
                    dW += self.L2(batch_size, dW=-1, where="backprop")
                db = np.sum(dZ, axis=1, keepdims=True) / batch_size   # dZ[last] / db[last]

                norm = LA.norm(dW)
                if norm > threshold:
                    dW = (threshold * dW) / norm
                norm = LA.norm(db)
                if norm > threshold:
                    db = (threshold * db) / norm

                grads_W.append(dW)
                grads_b.append(db)

                for l in reversed(range(self.L - 1)):  # calculate backprop
                    dA = np.dot(self.weights[l + 1].T, dZ)
                    if regularization == "dropout":
                        dA = self.dropout(masks, dA, layer=l, where="backprop")
                    dZ = dA * self.relu_backward(Zs[l])

                    dW = np.dot(dZ, activations[l].T)
                    if regularization == "L2":
                        dW += self.L2(batch_size, dW=l, where="backprop")
                    db = np.sum(dZ, axis=1, keepdims=True)

                    norm = LA.norm(dW)
                    if norm > threshold:
                        dW = (threshold * dW) / norm
                    norm = LA.norm(db)
                    if norm > threshold:
                        db = (threshold * db) / norm

                    grads_W.append(dW)
                    grads_b.append(db)

                for i in range(len(grads_W)):  # update parameters
                    if optimization == "momentum":
                        moments[-i - 1][0] = self.beta1 * moments[-i - 1][0] + (1 - self.beta1) * grads_W[i]
                        moments[-i - 1][1] = self.beta1 * moments[-i - 1][1] + (1 - self.beta1) * grads_b[i]
                        self.weights[-i - 1] -= l_rate * moments[-i - 1][0]
                        self.biases[-i - 1]  -= l_rate * moments[-i - 1][1]

                    elif optimization == "adam":
                        moments[-i - 1][0] = self.beta1 * moments[-i - 1][0] + (1 - self.beta1) * grads_W[i]
                        moments[-i - 1][1] = self.beta1 * moments[-i - 1][1] + (1 - self.beta1) * grads_b[i]

                        part1_W = moments[-i - 1][0] / (1 - self.beta1 ** 2)
                        part1_b = moments[-i - 1][1] / (1 - self.beta1 ** 2)

                        adams[-i - 1][0] = self.beta2 * adams[-i - 1][0] + (1 - self.beta2) * (grads_W[i] ** 2)
                        adams[-i - 1][1] = self.beta2 * adams[-i - 1][1] + (1 - self.beta2) * (grads_b[i] ** 2)

                        part2_W = adams[-i - 1][0] / (1 - self.beta2 ** 2)
                        part2_b = adams[-i - 1][1] / (1 - self.beta2 ** 2)

                        self.weights[-i - 1] -= l_rate * (part1_W / (np.sqrt(part2_W) + self.epsilon))
                        self.biases[-i - 1]  -= l_rate * (part1_b / (np.sqrt(part2_b) + self.epsilon))

                    else:
                        self.weights[-i - 1] -= l_rate * grads_W[i]
                        self.biases[-i - 1]  -= l_rate * grads_b[i]
            
            self.cost(costs, test_costs, m, train_X, train_Y, test_X, test_Y)
            accs.append(self.accuracy(train_X, train_Y))
            test_accs.append(self.accuracy(test_X, test_Y))
            print("{}/{}, train cost: {:.4f}, test cost: {:.4f}".format(x + 1, epoch, costs[-1], test_costs[-1]))

            indices = np.arange(int(m))
            np.random.shuffle(indices)
            train_X = train_X[:,indices]
            train_Y = train_Y[:,indices]

        title = "classes: " + str(train_Y.shape[0]) + ", l_rate: " + str(l_rate) + ", layers: " + self.layers.__str__()
        self.plot(costs, test_costs, title, "Loss", "cost")

        title = "Neural network accuracy"
        self.plot(accs, test_accs, title, "Accuracy", "accuracy")

        # print("last acc:", accs, test_accs)

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

    def predict(self, test_data):
        test_data = np.array(test_data)
        test_data = test_data.reshape(test_data.shape[0], 1)
        output = self.feedforward(test_data)

        print("output from nn: {} -> sum: {}".format(output, np.sum(output)))
        predictions = np.argmax(output)
        if output[predictions] < 0.5: 
            # predictions = -1
            pass
        return predictions

    def plot(self, arr1, arr2, title, ylabel, save, label1="train", label2="test", xlabel="Epoch"):
        plt.plot(arr1, label=label1)
        plt.plot(arr2, c="g", label=label2)
        plt.title(title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if self.reset_count != 0:
            save += str(self.reset_count)
        save += ".png"
        plt.savefig(save)
        plt.clf()

    def accuracy(self, test_data, Y):
        test_results = self.feedforward(test_data).T
        predict = 0
        Y = Y.T
        m = test_data.shape[1]
        for i in range(m):
            if np.argmax(test_results[i]) == np.argmax(Y[i]):
                predict += 1
        return (predict / m) * 100

    def cost(self, costs, test_costs, m, train_X, train_Y, test_X, test_Y):
        loss = self.config["loss"]
        regularization = self.config["regularization"]

        if loss == "cross_entropy":
            test_predict = self.feedforward(test_X)
            m_t = float(test_X.shape[1])
            test_cost = (np.sum(test_Y * np.log(test_predict) + (1. - test_Y) * np.log(1. - test_predict))) / -m_t
            test_costs.append(test_cost)    # L2 for test data also?

            train_predict = self.feedforward(train_X)
            cost = (np.sum(train_Y * np.log(train_predict) + (1. - train_Y) * np.log(1. - train_predict))) / -m
            if regularization == "L2":
                cost += self.L2(m)
            costs.append(cost)
        
        elif loss == "multi_label":
            test_predict = self.feedforward(test_X)
            m_t = float(test_X.shape[1])
            test_cost = (np.sum(test_Y * np.log(test_predict))) / -m_t
            # test_cost = (np.sum(np.nan_to_num(test_Y * np.log(test_predict) + (1. - test_Y) * np.log(1. - test_predict)))) / -m_t
            test_costs.append(test_cost)

            train_predict = self.feedforward(train_X)
            cost = (np.sum(train_Y * np.log(train_predict))) / -m
            # cost = (np.sum(np.nan_to_num(train_Y * np.log(train_predict) + (1. - train_Y) * np.log(1. - train_predict)))) / -m
            if regularization == "L2":
                cost += self.L2(m)
            costs.append(cost)

        elif loss == "mean_square":
            test_predict = self.feedforward(test_X)
            m_t = float(test_X.shape[1])
            test_cost = (np.sum(np.power(test_predict - test_Y, 2))) / m_t
            test_costs.append(test_cost)

            train_predict = self.feedforward(train_X)
            cost = (np.sum(np.power(train_predict - train_Y, 2))) / m
            if regularization == "L2":
                cost += self.L2(m)
            costs.append(cost)

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_backward(self, x):
        return (x > 0).astype(float)

    def softmax(self, data):
        exps = [np.exp(x) for x in data]
        sum_of_exps = np.sum(exps, axis=0)
        soft = np.array([x / sum_of_exps for x in exps])
        return soft

    def softmax_backward(self, x, Y):
        output = None
        for i in range(x.shape[1]):
            diag = np.sum(np.diag(x[:,i]) - np.dot(x[:,i], x[:,i].T), axis=0)
            if output is None:
                output = diag
            else:
                output = np.vstack((output, diag))
        return output.T

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_backward(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def save(self):
        np.save("weights", self.weights, allow_pickle=True)
        np.save("biases", self.biases, allow_pickle=True)
        np.savetxt("names.txt", self.names, delimiter=', ', fmt="%s")

    def reset_parameters(self, layers=None):
        if layers == None: 
            layers = self.layers
        else: 
            self.layers = layers
            self.L = len(layers) - 1
        self.reset_count += 1
        self.biases = np.array([np.zeros((x, 1)) for x in layers[1:]])
        self.weights = np.array([np.random.randn(layers[x], layers[x - 1]) * np.sqrt(2 / layers[x - 1]) for x in range(1, len(layers))])

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

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(self.layers[-1], activation="sigmoid")
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=adam, loss=loss_fn, metrics=["accuracy"])
        history = model.fit(x_train, y_train, epochs=5, verbose=2, batch_size=256)
        acc = model.evaluate(x_test, y_test)
        print("tf loss: {:.4f}, acc: %{:.2f}\n".format(acc[0], acc[1] * 100))

        plt.plot(history.history["loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig("TFcost.png")
        plt.clf()

        plt.plot(history.history["accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.savefig("TFaccuracy.png")
        plt.clf()

if __name__ == '__main__':
    DoodleClassifier()
