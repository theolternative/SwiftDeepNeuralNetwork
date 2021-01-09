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
- Define a train set X, Y where X is a `n x m` input matrix  containing `m` examples of `n` features each and a `1 x m` Y output matrix of labels. All values must be `Double`
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

### Initialize
You can initialize a new model as

```swift
let dnn = DeepNeuralNetwork( layerDimensions: [Double], X: Matrix, Y: Matrix)
```
where layerDimensions is an array of  `Int` specyfing number of hidden units per layer, X is the features matrix and Y is the labels matrix

### Setting up training
Following hyperparameters can be set before training

#### Learning rate
Name:  `learning_rate`
Type: `Double`
Default value: 0.0075

Learning rate determines the overall velocity of gradient descent. Low values (<0.001) can lead to slow performance but better accuracy while high values can lead to better performance but worse accuracy or instability.

```swift
dnn.learning_rate = 0.15
```

**Hint: start with low values (0.0001) and increase by an order if too slow**

#### Number of epochs
Name:  `num_iterations`
Type: `Int`
Default value: 2500

This hyperparameter controls how many iterations are performed by gradient descent. Low values can lead to better performance but worse accuracy while high values lead to better accuracy with performance as a trade-off.

```swift
dnn.num_iterations = 2500
```

#### Initialization type
Name:  `weigth_init_type`
Type: `Initialization`
Default value:  `random`

Weights' matrices must be initialized with random values in order for the training to work. Following types are available:

| Value         | Description                                                                                             |
|-------------|---------------------------------------------------------------------------------|
| `.zeros`    | Values set to 0 (only for debug)                                                              |
| `.random`  | Random values in range -1.0....1.0                                                        |
| `.xavier`  | Xavier initialization                                                                                  |
| `.he`          | He initialization                                                                                       |
| `.custom`  | Multiply random values -1.0...1.0 by factor  `custom_weight_factor` |

```swift
dnn.weigth_init_type = .he
```

#### Regularization

Two types of regularization in order to reduce variance between train and test sets, are provided: L2 and Dropout 

##### L2 Regularization
Name:  `λ`
Type: `Double`
Default value:  0.0

If a value greater than 0.0 is set for  `λ` it will be applied.

```swift
dnn.λ = 0.7
```
##### Dropout regularization
Name:  `keep_prob`
Type: `Double`
Default value:  1.0

Dropout Regularization randomly shuts off some elements in weights matrices so that a percentage of `keep_prob` is not zero.
If a value less than 1.0 is set for  `keep_prob`, dropout regularization will be applied.

```swift
dnn.keep_prob = 0.85 // 15% elements of weights matrices will be 0
```

#### Optimization type
Name:  `optimization_type`
Type: `Optimization`
Default value:  `gradientdescent`

Following choices are available

| Value                            | Description                                             |
|---------------------------|----------------------------------------------|
| `.gradientdescent`    | Gradient descent                                   |
| `.momentum`                  | Gradient descent with momentum        |
| `.adam`                          | Adam optimization method                   |

For `.momentum` optmization method `β` factor can be set (deafult is 0.1)
For `.adam` optmization method `β1` and `β2` factors can be set (deafults are 0.9 and 0.999)

```swift
dnn.optimization_type = .adam
dnn.β1 = 0.91
dnn.β2 = 0.98
```

#### Batch type
Name:  `batch_type`
Type: `Batch`
Default value: `.batch`

Batch type determines how many examples are computed all at a time.

| Value                            | Description                                                     |
|---------------------------|----------------------------------------------------|
| `.batch`                        | All examples at a time                                    |
| `.minibatch`                | `mini_batch_size`  examples at a time        |
| `.stochastic`              | 1 example at a time                                        |

```swift
dnn.batch_type = .minibatch
dnn.mini_batch_size = 64
```
**Hint: Stochastic gradient descent is very slow but can lead to better accuracy after fewer epochs. When examples tend to become many (>1000) training times become longer and Minibatch approach should be the default choice**

