# Doodle Classifier
A neural network classifier. *See test/ for images*
## Specifications: ##
* Interactive GUI in Tkinter, allows to the user choose various settings
* Neural network can run up to 4 layers (without changing the source code)
* Supports L2, adam and dropout
* Plots cost of train and test data and saves.
* Predicts doodles with Google Quick, Draw Dataset
* Shows selected images from data
* Saves biases and weights of the neural network or loads
* If predict false, train itself with drawn doodle
* If no data present, only options is load a saved neural network
## Dependencies: ##
* Pillow
* Numpy
* Matplotlib
* Tensorflow (If not installed, just remove it)
* Tested on Python 3.7.6 x64
## Run: ##
`$ python DoodleClassifier.py`
## Benchmarks: ##
1. `tf.keras.datasets.mnist`:
   - Neural Net: [28 \* 28, 64, 32, 10], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->sigmoid_: %96.27
2. `tf.keras.datasets.fashion_mnist`:
   - Neural Net: [28 \* 28, 64, 32, 10], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->sigmoid_: %85.78
3. Google Quick, Draw Dataset (3 images, 5000 examples per):
   - Neural Net: [28 \* 28, 64, 32, 10], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->sigmoid_: %89.40
## Usage: ##
- **Create the neural net:**
```python
NeuralNetwork(layers, names, l_rate=0.01)
```
- **Train:**
```python
NeuralNetwork.train(x_train, y_train, x_test, y_test, epoch=5, batch_size=256, optimizer="adam")
```
- **Calculate accuracy:**
```python
NeuralNetwork.accuracy(x_test, y_test)
```
### Bugs: ###
* plit.close() also closes Tkinter. If plt not closed, program runs at background. (see [#13470](https://github.com/matplotlib/matplotlib/issues/13470))
* Softmax at last layer doesn't work properly. In NeuralNetwork.py, I accidently make this error `dZ = predict - Y / mini`, instead of `dZ = (predict - Y) / mini`. This kind of works but the cost is high. When it is corrected, softmax raises "RuntimeWarning: overflow encountered in exp"
* When program predicts false, if you hit false button and do not select true image, that segment does not go away.
* PILL.ImageGrab support OS X and Windows only

_Please let me know if there is something wrong_
