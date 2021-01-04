# SwiftDeepNeuralNetwork


SwiftDeepNeuralNetwork is a Swift library based on [SwiftMatrix](https://github.com/theolternative/SwiftMatrix) library and it aims to provide a step by step guide to Deep Neural Network implementation. It's largely based on code developed in [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome) course by Andrew Ng on Coursera.

**SwiftDeepNeuralNetwork has been developed in order to show all necessary steps to write code for Deep Learning in the easiest possible way**


---

## Installation

_Using XCode v.12 and upper add a package dependancy by selecting File > Swift Packages > Add Package Dependancy ... and entering following URL https://github.com/theolternative/SwiftDeepNeuralNetwork.git_

SwiftDeepNeuralNetwork uses Swift 5 and Accelerate Framework and SwiftMatrix as a dependancy.

## License

SwiftDeepNeuralNetwork is available under the MIT license. See the LICENSE file for more info.

## Usage

You can check Tests for examples. Currently DNN supports only binary classification.
Basically
- Define a train set X, Y where X is a `n x m` input matrix  containing `m` examples of `n` features each and a `1 x m` Y output matrix
- Create an instance of  `DeepNeuralNetwork` passing an array of `Int` where the first element is the number of features `n`, the last one is always 1 (binary classification) while the others specify number of units per layer. [2000, 20, 5, 1] means that there are 1 input layer of 2000 features (and must match with number of rows of X), 2 hidden layers of 20 and 5 units each and and outpute layer.
- Set `learning_rate` and `num_iterations`
- Train your model by calling function `L_layer_model_train()`
- Make predictions on test set with `L_layer_model_test( X_test, Y_test)` where X_test must have size `n x p` and Y_test size  `1 x p` where p >= 1 

```swift
// XOR example
let X = Matrix([ [0,0], [0,1], [1,0], [1,1] ])′
let Y = Matrix([[ 0, 1, 1, 0 ]])
let dnn = DeepNeuralNetwork(layerDimensions: [2, 2, 1], X: X, Y: Y)
dnn.learning_rate = 0.15
dnn.num_iterations = 1000
let (accuracy, costs) = dnn.L_layer_model_train(nil)
```
### Notes
Currently  regularization is not supported and results may heavily change depending on initial values of weigths' matrices. Specifically XOR test may succeed or fail unpredictably.
In order to avoid failure proper weigths' matrices can be passed.
