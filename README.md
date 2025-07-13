# Doodle Classifier
A neural network classifier. *See test/ for images*
## Specifications: ##
* Interactive GUI in Tkinter, allows to the user choose various settings
* Neural network can run up to 4 layers (without changing the source code)
* Supports L2, Dropout regularizations and Adam, momentum, mini batch optimizers and various loss function
* Plots cost and accuracy of train and test data and saves.
* Resets paramenters according to new layers or old one.
* Predicts doodles with Google Quick, Draw Dataset
* Shows selected images from data
* Saves biases and weights of the neural network or loads
* If predict false, train itself with drawn doodle
* If no data present, only options is load a saved neural network
## Dependencies: ##
* Pillow
* Numpy
* Matplotlib
* Tested on Python 3.7.6 x64
## Run: ##
`$ python DoodleClassifier.py`
<table>
    <tr>
        <td align="center">
            <img src="https://github.com/ssduman/doodle-classifier/blob/master/test/doodle.gif" alt="home-page" width="384" height="450">
            <br />
            <i> demo </i>
        </td>
    </tr>
</table>

## Benchmarks: ##
1. `tf.keras.datasets.mnist`:
   - Neural Net: [28 \* 28, 64, 32, 10], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->softmax_: **%95.19**
2. `tf.keras.datasets.fashion_mnist`:
   - Neural Net: [28 \* 28, 64, 32, 10], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->softmax_: **%86.82**
3. Google Quick, Draw Dataset (3 images, 5000 examples per):
   - Neural Net: [28 \* 28, 64, 32, 3], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->softmax_: **%90.00**
4. Google Quick, Draw Dataset (5 images, 5000 examples per):
   - Neural Net: [28 \* 28, 64, 32, 5], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->softmax_: **%81.44**
4. Google Quick, Draw Dataset (10 images, 5000 examples per):
   - Neural Net: [28 \* 28, 64, 32, 10], _batch_size_=256, _lr_=0.01, _adam_, _relu->relu->softmax_: **%77.34**
## Usage: ##
- **Create the neural net:**
```python
NeuralNetwork(layers)
```
- **Train:**
```python
# defaults, change or delete any of them
config = {
   "l_rate" : 0.01, 
   "epoch" : 5, 
   "batch_size" : 256, 
   "loss" : "multi_label",     # "cross_entropy", "multi_label", "mean_square"
   "optimization" : "adam",    # "adam", "momentum"
   "regularization" : "none"   # "dropout", "L2"
}

NeuralNetwork.train(x_train, y_train, x_test, y_test, config)
```
- **Get MNIST:**
```python
x_train, y_train, x_test, y_test = DoodleClassifier.test_mnist("mnist") # "mnist", "fashion_mnist"
```
- **Calculate accuracy:**
```python
NeuralNetwork.accuracy(x_test, y_test)
```
- **Reset parameters:**
```python
NeuralNetwork.reset_parameters(new_layer)
```
### Bugs: ###
* Calling plt.close() also closes Tkinter. If Matplotlib is not closed, program runs at background. (see [#13470](https://github.com/matplotlib/matplotlib/issues/13470))
* When program predicts false, if you hit false button and do not select true image, that segment does not go away.
* PILL.ImageGrab support OS X and Windows only.

_Please let me know if there is something wrong. CNN is under construction._
