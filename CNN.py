from numba import jit
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pickle
import time
import cProfile
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

class CNN(object):
    def __init__(self, layers=[3, 8, 16], fc_layers=[576, 32, 10]):
        self.images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
        self.labels = np.zeros(shape=[10, 50000], dtype=int)
        self.test_images = np.zeros(shape=[10000, 32, 32, 3], dtype=float)
        self.test_labels = None
        self.load_data()
        # self.visualize_data(self.images, self.labels)
        # self.lena(filename="valve.png", Filter=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])) # sobel

        self.batch_size = 10000
        self.layers = layers[1:]
        self.fc_layers = fc_layers
        self.L = len(layers) - 1
        self.filters = np.array([np.random.rand(layers[x + 1], 3, 3, layers[x]) / 9 for x in range(len(layers) - 1)])

        self.biases = np.array([np.zeros((x, 1)) for x in fc_layers[1:]])
        self.weights = np.array(
            [np.random.randn(fc_layers[x], fc_layers[x - 1]) * np.sqrt(2 / fc_layers[x - 1]) 
            for x in range(1, len(fc_layers))]
        )
        self.l_rate = 0.01
        self.Images = []
        self.As = []
        self.Zs = []

        self.epoch = 5

        self.m = self.images.shape[0]
    
    def train(self):
        for i in range(self.epoch):

            start = time.time()
            for y in range(0, int(self.m), self.batch_size):
                if y + self.batch_size >= self.m:
                    y = int(self.m - self.batch_size - 1)
                data = self.images[y:y + self.batch_size, :]
                labels = self.labels[:, y:y + self.batch_size]
                
                output = self.conv(data, self.batch_size)
                grad = self.fully_connected_backprop(output, labels)
            print("epoch: {}".format(time.time() - start))

            start = time.time()
            predict = self.feedfoward(self.test_images)
            print("feedf: {}".format(time.time() - start))

            # print("predict: {}".format(predict.shape))
            # print("  shape: {}, {}".format(predict[:, 1].shape, self.test_labels[:, 1].shape))
            
            cost = CNN.cost(predict, self.test_labels)
            acc = CNN.accuracy(predict, self.test_labels)
            print("{}/{}, cost: {:.4f}, acc: %{:.2f}".format(i + 1, self.epoch, cost, acc))

    @staticmethod
    @jit(nopython=True)
    def iterate(output, data, f, h, w, batch_size):
        nf = f.shape[0]
        for i in range(h - 2):
            for j in range(w - 2):
                for x in range(batch_size):
                    # print("1: {}".format(output[x, i, j].shape))                    # (8,)
                    # print("2: {}".format(data[x, i:i+3, j:j+3].shape))              # (3, 3, 3)
                    # print("3: {}".format(np.sum(data[x, i:i+3, j:j+3] * f, axis=(1, 2, 3)).shape))  # (8,)
                    # print("f:", f.shape)                                            # (8, 3, 3, 3)
                    
                    s = np.zeros((nf,))
                    for c in range(nf):
                        # print("0: {}".format(np.sum(data[x, i:i+3, j:j+3] * f[c]).shape))   # ()
                        # print("1: {}".format(f[c].shape))                                   # (3, 3, 3)
                        # print("2: {}".format(s.shape))                                      # (8,)
                        s[c] = np.sum(data[x, i:i+3, j:j+3] * f[c])
                    output[x, i, j] = s

    def conv(self, data, batch_size):
        for l in range(self.L):
            f = self.filters[l].shape[0]
            h = data.shape[1]
            w = data.shape[2]
            
            output = np.zeros((batch_size, h - 2, w - 2, self.layers[l]))   
            CNN.iterate(output, data, self.filters[l], h, w, batch_size)
            data = self.relu(CNN.pooling(output, batch_size, h - 2, w - 2, self.layers[l]))  

        data = data.reshape(-1, batch_size)
        return self.fully_connected(data)

        # --training--
        # for l in range(self.L):
            # grad = self.pooling_backprop(grad)
            # grad = self.conv_backrop(grad)

    @staticmethod
    @jit(nopython=True)
    def pooling(array, batch_size, h, w, nc, mode="maxpool", f=2, pad=0, stride=2):
        h_m = int((h + 2 * pad - f) / stride + 1)
        w_m = int((w + 2 * pad - f) / stride + 1)

        pool = np.zeros((batch_size, h_m, w_m, nc))

        for i in range(0, h_m + 1, stride):
            for j in range(0, w_m + 1, stride):
                for t in range(batch_size):
                    x = int(i / stride)
                    y = int(j / stride)
                    for c in range(nc):
                        if mode == "maxpool":
                            pool[c, t, x, y] = np.max(array[t, i:i+f, j:j+f, c])
                        elif mode == "mimpool":
                            pool[c, t, x, y] = np.min(array[t, i:i+f, j:j+f, c])
                        elif mode == "averagepool":
                            pool[c, t, x, y] = np.sum(array[t, i:i+f, j:j+f, c]) / (f * f)

        return pool

    def fully_connected(self, data):
        self.As = [data]
        self.Zs = []

        A = data
        L = len(self.fc_layers) - 1
        for l in range(L - 1):
            z = np.dot(self.weights[l], A) + self.biases[l]
            self.Zs.append(z)
            A = self.relu(z)
            self.As.append(A)
        z = np.dot(self.weights[L - 1], A) + self.biases[L - 1]
        self.Zs.append(z)
        A = self.softmax(z - np.max(z))
        self.As.append(A)
        return A

    def feedfoward(self, data):
        return self.conv(data, data.shape[0])
    
    @staticmethod
    @jit(nopython=True)
    def cost(predict, Y):
        return (np.sum(Y * np.log(predict))) / -predict.shape[1]

    @staticmethod
    @jit(nopython=True)
    def accuracy(predict, Y):
        acc = 0
        L = predict.shape[1]
        for i in range(L):
            if np.argmax(predict[:, i]) == np.argmax(Y[:, i]):
                acc += 1
        
        return (acc / L) * 100

    def cost_derivative(self, predict, Y):
        return -Y * (1 / predict)

    def fully_connected_backprop(self, predict, Y):
        threshold = 0.5

        grads_W = []
        grads_b = []

        batch_size = Y.shape[0]
        dZ = (predict - Y) / batch_size

        dW = np.dot(dZ, self.As[-2].T) / batch_size         # dl_dy * dy_dz * dz_dw
        db = np.sum(dZ, axis=1, keepdims=True) / batch_size # dl_dy * dy_dz * dz_db

        norm = LA.norm(dW)
        if norm > threshold:
            dW = (threshold * dW) / norm
        norm = LA.norm(db)
        if norm > threshold:
            db = (threshold * db) / norm

        grads_W.append(dW)
        grads_b.append(db)

        for l in reversed(range(len(self.fc_layers) - 2)):
            dA = np.dot(self.weights[l + 1].T, dZ)
            dZ = dA * self.relu_backward(self.Zs[l])

            dW = np.dot(dZ, self.As[l].T)
            db = np.sum(dZ, axis=1, keepdims=True)

            norm = LA.norm(dW)
            if norm > threshold:
                dW = (threshold * dW) / norm
            norm = LA.norm(db)
            if norm > threshold:
                db = (threshold * db) / norm

            grads_W.append(dW)
            grads_b.append(db)
        
        for i in range(len(grads_W)):
            self.weights[-i - 1] -= self.l_rate * grads_W[i]
            self.biases[-i - 1]  -= self.l_rate * grads_b[i]

        return dZ
        
    def pooling_backprop(self, grad, batch_size):
        out = np.zeros(self.Images[-1].shape)

        for i in range(0, out.shape[1]):
            for j in range(0, out.shape[2]):
                for t in range(batch_size):

                    h, w, n = grad[t, i:i+2, j:j+2].shape
                    amax = np.amax(grad[t, i:i+2, j:j+2], axis=(0, 1))

                    for i2 in range(h):
                        for j2 in range(w):
                            for f2 in range(2):
                                if grad[t, i2, j2, f2] == amax[f2]:
                                    out[i * 2 + i2, j * 2 + j2, f2] = grad[i, j, f2]
        
        return out
  
    def conv_backrop(self, grad, batch_size, l):
        out = np.zeros(self.filters.shape)

        for i in range(0, out.shape[1]):
            for j in range(0, out.shape[2]):
                for t in range(batch_size):
                    for f in range(self.num_filters):
                        out[f] += grad[i, j, f] * self.Images[l]

        self.filters -= self.l_rate * out

        return out # loss gradient should return
        
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
        # output = None
        # for i in range(x.shape[1]):
        #     diag = np.sum(np.diag(x[:,i]) - np.dot(x[:,i], x[:,i].T), axis=0)
        #     if output is None:
        #         output = diag
        #     else:
        #         output = np.vstack((output, diag))
        # return output.T
        return x * (1 - x)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_backward(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def load_data(self):
        begin = 0
        for i in range(5):
            with open("data/cifar-10-python/data_batch_" + str(i + 1), "rb") as f:
                data = pickle.load(f, encoding="latin1")
                image = data["data"]
                image = image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
                label = data["labels"]
                Y = np.zeros((10000, 10))
                Y[np.arange(10000), label] = 1
                end = begin + len(image)
                self.images[begin:end, :] = image / 255. - 0.5
                self.labels[:, begin:end] = Y.T
                begin = end
        
        with open("data/cifar-10-python/test_batch", "rb") as f:
            data = pickle.load(f, encoding="latin1")
            image = data["data"]
            image = image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
            label = data["labels"]
            self.test_images = image / 255. - 0.5
            Y = np.zeros((10000, 10))
            Y[np.arange(10000), label] = 1
            self.test_labels = Y.T
    
    def visualize_data(self, data, labels, row=4, col=4):
        fig, ax = plt.subplots(row, col, figsize=(5, 5))
        fig.tight_layout()

        classes = { 0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 
                    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck" }
        r = np.random.randint(data.shape[0] - row * col)
        for i in range(row):
            for j in range(col):
                index = r + j + col * i
                ax[i][j].imshow(((data[index, :] + 0.5 ) * 255.).astype("uint8"))
                ax[i][j].set(title=classes[np.argmax(labels[:, index])])
                ax[i][j].axis(False)
        
        plt.show()

    def lena(self, filename="lena.jpg", Filter=None, RGB=True, gray=True):
        """
        Description:
        Applies sobel filter for default arguments
        Reference: https://en.wikipedia.org/wiki/Sobel_operator
        If any filter is given, then sobel operator will not apply to the image.
        Sobel filter or any other filters above works fine

        Arguments:
        Filter: edge, identity, blur or any other filter like:
            sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            edge1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            edge2 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
            identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            box_blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
            gaussian_blur_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            gaussian_blur_5x5 = np.array([
                    [1,  4,  6,  4, 1], 
                    [4, 16, 24, 16, 4], 
                    [6, 24, 36, 24, 6], 
                    [4, 16, 24, 16, 4],
                    [1,  4,  6,  4, 1]]) / 256
        RGB: input image is form of RGB or grayscale
        gray: output image is form of RGB or grayscale

        Bugs:
        Differences PNG and JPEG images (Found by testing different images, could be wrong):
        plt.imshow arguments vmin and vmax differs. For PNG, vmax is 1, for JPEG vmax is 255
        plt.imshow argument image takes astype("uint8") for PNG bu not for JPEG
        If any of them wrong, displayed image either full black or full white or raises
        "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or 
        [0..255] for integers)."

        PIL implemetation:
        img = Image.open(filename).convert("L")
        edge = img.filter(ImageFilter.FIND_EDGES)
        edge.show()
        """
        PNGorJPG = filename.find(".png")

        img = imread(filename)
        if not RGB:
            img = self.rgb2gray(img)

        h, w = img.shape[0], img.shape[1]
        img = self.gaussian_filter(img, h, w, RGB=RGB)  # remove noise
        img = self.convolution_filter(img, h, w, Filter=Filter, RGB=RGB)    # apply sobel filter for RGB or grayscale images
        
        if gray:
            if RGB:
                img = self.rgb2gray(img)
            if PNGorJPG != -1:  # PNG image is given
                plt.imshow(img, cmap="gray", vmin=0, vmax=1)    # vmin,vmax: 0-1 or 0-255?
            if PNGorJPG == -1:  # JPG image is given
                plt.imshow(img, cmap="gray", vmin=0, vmax=255)  # vmin,vmax: 0-1 or 0-255?
        else:
            if PNGorJPG != -1:  # PNG image is given
                plt.imshow(img)
            if PNGorJPG == -1:  # JPG image is given
                plt.imshow(img.astype("uint8"))
        plt.axis(False)
        plt.show()

    def convolution_filter(self, img, h, w, f=3, pad=1, stride=1, Filter=None, RGB=True, remove_noise=False):
        if Filter is not None:
            f = Filter.shape[0]
        h_c = int((h + 2 * pad - f) / stride + 1)
        w_c = int((w + 2 * pad - f) / stride + 1)

        if RGB:
            Rc = np.zeros((h_c, w_c))
            Gc = np.zeros((h_c, w_c))
            Bc = np.zeros((h_c, w_c))

            R = np.pad(img[:,:,0], ((pad, pad), (pad, pad)), "constant", constant_values=0)
            G = np.pad(img[:,:,1], ((pad, pad), (pad, pad)), "constant", constant_values=0)
            B = np.pad(img[:,:,2], ((pad, pad), (pad, pad)), "constant", constant_values=0)
        else:
            Grayc = np.zeros((h_c, w_c))
            Gray = np.pad(img[:,:], ((pad, pad), (pad, pad)), "constant", constant_values=0)

        if not remove_noise:
            Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        for i in range(h_c):
            for j in range(w_c):
                if RGB and remove_noise or Filter is not None and RGB:
                    Rc[i, j] = np.sum(R[i:i+f, j:j+f] * Filter)
                    Gc[i, j] = np.sum(G[i:i+f, j:j+f] * Filter)
                    Bc[i, j] = np.sum(B[i:i+f, j:j+f] * Filter)

                elif RGB and not remove_noise:
                    r = R[i:i+f, j:j+f]
                    S1 = np.sum((Gx * r))
                    S2 = np.sum((Gy * r))
                    Rc[i, j] = np.sqrt(np.power(S1, 2) + np.power(S2, 2))

                    g = G[i:i+f, j:j+f]
                    S1 = np.sum((Gx * g))
                    S2 = np.sum((Gy * g))
                    Gc[i, j] = np.sqrt(np.power(S1, 2) + np.power(S2, 2))

                    b = B[i:i+f, j:j+f]
                    S1 = np.sum((Gx * b))
                    S2 = np.sum((Gy * b))
                    Bc[i, j] = np.sqrt(np.power(S1, 2) + np.power(S2, 2))

                elif not RGB and remove_noise or Filter is not None and not RGB:
                    Grayc[i, j] = np.sum(Gray[i:i+f, j:j+f] * Filter)

                elif not RGB and not remove_noise:
                    gray = Gray[i:i+f, j:j+f]
                    S1 = np.sum((Gx * gray))
                    S2 = np.sum((Gy * gray))
                    Grayc[i, j] = np.sqrt(np.power(S1, 2) + np.power(S2, 2))
        
        if RGB:
            return np.dstack((Rc, Gc, Bc))
        else:
            return Grayc
    
    def gaussian_filter(self, img, h, w, RGB=True):
        Filter = np.array([
            [2,  4,  5,  4, 2], 
            [4,  9, 12,  9, 4], 
            [5, 12, 15, 12, 5], 
            [4,  9, 12,  9, 4], 
            [2,  4,  5,  4, 2], 
        ]) / 159

        return self.convolution_filter(img, h, w, f=5, pad=2, Filter=Filter, RGB=RGB, remove_noise=True)
    
    def normalize(self, img):
        Max = img.max()
        Min = img.min()

        return (img - Min) / (Max - Min)

    def rgb2gray(self, img):
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    def tf(self):
        """
        Code from: https://keras.io/examples/cifar10_cnn/
        """
        batch_size = 32
        num_classes = 10
        epochs = 100
        num_predictions = 20

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(num_classes))
        model.add(tf.keras.layers.Activation('softmax'))

        opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
              
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

if __name__ == '__main__':
    cnn = CNN()
    cnn.tf()
    # cnn.train()
    # cProfile.run("cnn.train()")
