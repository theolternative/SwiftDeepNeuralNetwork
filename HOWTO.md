# How To
This file is intended as a super synopsis of all basic concepts in order to develop Machine Learning projects. Specifically it's a simplicistic and condensed summary (did I say it's very basic?) of topics treated in [Deep Learning Specialization Course](https://www.coursera.org/specializations/deep-learning).

## Structuring Machine Learning Projects

### Neural networks
A neural network (*NN*) finds a *mapping function* between an input feature matrix `X` to an ouput label matrix `Y`. A NN comprises several **layers** each composed of different **units**. The first  layer is the *input* layer, the last one is the *output layer*. Middle layers are called *hidden*. Depending on the number of hidden layers, a NN can be *shallow* (only a few) or *deep* (many). 
In a typical *supervised learning* problem, first you *train* a NN with labelled examples, then you *predict*. Input data can be *structured* (each input feature has a well-defined meaning, i.e. temperature, humidity, wind velocity for weather forecasts) or *unstructued* (i.e. images, audio samples, etc.).


#### Notation
`L` number of layers including input and output layers
`l` index of current layer ranging from 1 to `L`
`X` input feature matrix of dimension n x m (rows by columns) where n is the number of features and m is the number of examples
`Y` ouput label matrix of dimension r x m (rows by columns) where r is the number of label classes (1 for binary classification problems) and m is the number of examples
`W(l)` weigths' matrix of layer `l`
`b(l)` biases' matrix of layer `l`
`Z(l)` input matrix of layer `l`
`A(l)` activation matrix of layer `l` computed as `g(Z(l))` where g(x) is the activation function
`Y_hat(L)` prediction output matrix

#### Activation functions
Activation functions are needed in order to introduce non-linearity in neural networks, otherwise relationships between layers would be linear just like in *linear regression*:  
![activationfunction](http://latex.codecogs.com/png.latex?\dpi{110}%20g^{[l]}(z)\text{%20activation%20function%20of%20layer%20l,%20}\mathbf{A^{[l]}}=g^{[l]}(\mathbf{W^{l}A^{[l-1]]}+B^{[l]}}))

**Sigmoid**
Sigmoid function is used for output layer in *logistic regression*. When input parameter z becomes larger, output tends to 1.0, while when input parameter z becomes very small, output tends to 0.0  
![sigmoid](https://latex.codecogs.com/svg.latex?\sigma%20(z)%20=%20\frac{1}{1+e^{-z}})  
![sigmoidder](https://latex.codecogs.com/svg.latex?{\sigma(z)}%27=\sigma(z)(1-\sigma(z)))  

**Rectified LineaR Unit (RELU)**
Generally used for hidden layers.  
![relu](https://latex.codecogs.com/svg.latex?g(z)=max(0,z))  
![reluder](http://latex.codecogs.com/svg.latex?g%27(z)=\begin{cases}0%20&%20\text{%20if%20}%20z%3C0%20\\1%20&%20\text{%20if%20}%20z%3E0%20\end{cases})  

**Leaky Rectified LineaR Unit (RELU)**  
![leakyrelu](https://latex.codecogs.com/svg.latex?g(z)=max(0.01,z))  
![leakyreluder](http://latex.codecogs.com/svg.latex?g%27(z)=\begin{cases}0.01%20&%20\text{%20if%20}%20z%3C0%20\\1%20&%20\text{%20if%20}%20z%3E0%20\end{cases})  

**Tanh**
Can be used for *logistic regression* but generally sigmoid is preferred.  
![tanh](https://latex.codecogs.com/svg.latex?tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}})  
![tanhder](https://latex.codecogs.com/svg.latex?{tanh}%27(z)=1-(tanh(z))^{2})  

#### Cost functions

**Binary classification**
*Cost function* `J(W,b)` is defined as the average of  sum of *loss function* `L(y_hat(i),y(i))` computed for each example with i ranging from 1 to m.  
![costfunction](https://latex.codecogs.com/svg.latex?J(\mathbf{W},\boldsymbol{\mathbf{}b})=\frac{1}{m}\sum_{n=1}^m%20L(\hat{y}^{(i)},y^{(i)}))  
![lossfunction](https://latex.codecogs.com/svg.latex?L(\hat{y}^{(i)},y^{(i)})%20=%20-y^{(i)}%20\log\hat{y}^{(i)}+(1-y^{(i)})\log(1-\hat{y}^{(i)}))  
![costfunction](https://latex.codecogs.com/svg.latex?J(\mathbf{W},\boldsymbol{\mathbf{}b})=-\frac{1}{m}%20(\mathbf%20{Y}%20\log\mathbf%20{\hat{Y}}+(1-\mathbf%20{Y})\log(1-\mathbf%20{\hat{Y}})))
when y(i) is 1, cost function tends to 0 if y_hat(i) (*prediction*) is also near 1,  on the other hand if y(i)=0, cost function tends to 0 if y_hat(i) is near 0. This way cost function is minimized when predictions match output values. This is *convex function* which means that you can *learn* parameters `W` and `b` by finding the global mimimum via an *optimizer* 

Derivative of loss function used in computational graph is:  
![binarylossder](https://latex.codecogs.com/svg.latex?\frac{\partial%20\mathbf{L}}{\partial%20\mathbf{\hat{Y}}}=-\frac{\mathbf{Y}}{\mathbf{\hat{Y}}}+\frac{1-\mathbf{Y}}{1-\mathbf{\hat{Y}}})

#### Optimizer
**Gradient descent**
Parameters are adjusted after each iteration for each layer `l` with following formulas:  
![Wupdate](https://latex.codecogs.com/svg.latex?\mathbf{W^{(l)}}=\mathbf{W^{(l)}}-\alpha\frac{\partial\mathbf{L}}{\partial\mathbf{W^{(l)}}})  
![bupdate](https://latex.codecogs.com/svg.latex?\mathbf{b^{(l)}}=\mathbf{b^{(l)}}-\alpha\frac{\partial\mathbf{L}}{\partial\mathbf{b^{(l)}}})  

Derivatives for each layer are computed via a *computational graph* and the following derivation rule:  
![derrule](https://latex.codecogs.com/svg.latex?\frac{\partial%20z(v(u(x)))}{\partial%20x}=\frac{\partial%20z}{\partial%20v}%20\frac{\partial%20v}{\partial%20u}\frac{\partial%20u}{\partial%20x})

For last layer `L` parameters are updated as:  
![LupdateW](https://latex.codecogs.com/svg.latex?\mathbf{W^{[L]}}=\mathbf{W^{[L]}}-\alpha%20\frac{\partial%20\mathbf{L}}{\partial%20\mathbf{W^{[L]}}}\newline%20\frac{\partial%20\mathbf{L}}{\partial%20\mathbf{W^{[L]}}}=\frac{\partial%20\mathbf{L}}{\partial%20\mathbf{\hat{Y}}}*\frac{\partial%20\mathbf{\hat{Y}}}{\partial%20\mathbf{W^{[L]}}}\newline%20\newline%20\mathbf{\hat{Y}}=g(z(\mathbf{W^{[L]}%20A^{[L-1]}+B^{[L]}}))\newline%20%20\rightarrow%20\frac{\partial%20\mathbf{\hat{Y}}}{\partial%20\mathbf{W^{[L]}}}%20=%20\mathbf{A^{[L-1]}}%20*%20\frac{\partial%20g(z)}{\partial%20z})  
![BupdateW](https://latex.codecogs.com/svg.latex?\mathbf{B^{[L]}}=\mathbf{B^{[L]}}-\alpha%20\frac{\partial%20\mathbf{L}}{\partial%20\mathbf{B^{[L]}}}\newline%20\frac{\partial%20\mathbf{L}}{\partial%20\mathbf{B^{[L]}}}=\frac{\partial%20\mathbf{L}}{\partial%20\mathbf{\hat{Y}}}*\frac{\partial%20\mathbf{\hat{Y}}}{\partial%20\mathbf{B^{[L]}}}\newline%20\newline%20\mathbf{\hat{Y}}=g(z(\mathbf{W^{[L]}%20A^{[L-1]}+B^{[L]}}))\newline%20%20\rightarrow%20\frac{\partial%20\mathbf{\hat{Y}}}{\partial%20\mathbf{B^{[L]}}}%20=%20\frac{\partial%20g(z)}{\partial%20z})  


#### Main steps
1) Initialize *weights matrices* `W(l)` of each layer using random numbers or other commonly used methods (i.e Xavier or He)
2) Feed forward through each layer (*forward propagation*) and compute `Z(l) = W(l) a(l-1) + b(l)` and apply activation function `g()` (generally RELU for hidden layers and Sigmoid for output layer in binary classification or Softmax in multi-class classification problems)
3) Compute cost function and prediction accuracy on training set. 
4) Adjust parameters `W` and `b` via gradient descrent and *back propagation* up to the first layer
5) Iterate through points 2-4 till a cost < `accepterd_threshold`, accuracy > `accepted_accuracy` or iterations = `num_epochs`

### Definitions
*Accuracy* is the ratio between correct vs total number of predictions
*Error* is 1.0-Accuracy
*Bayes error* is the lowest possible prediction error. Nor humans nor AI can achieve lower values.
*Human error* is the mean human prediction error. Human error >= Bayes error. It's generally used as a benchmark for NN error
*Bias* is the error on training set
*Avoidable bias* is the difference between error on training set and Bayes error (or human error if more meaningful)
*Variance* is the difference between error on test set and error on training set
*Data mismatch error* is the error due to data coming from different distributions
*Underfit* occurs when accuracy on training set is too low
*Overfit* may occur when accuracy on training set is good but error on dev set is high.

### Deep Learning Cycle
* **Define your goal**: are you trying to classify a set of images for one food (**binary classification** using *Logistic regression*), for one food out of a selection (**multi-class classification** using *Softmax function*) or for multiple foods at a time (**multi-task classification** using multiple *Logistic regressions*)?
*  **Pick a metric** (i.e. classification error). In case of multiple criteria, create a unique evaluation (i.e. harmonic mean = 2 / (1/E1+1/E2) where E1 can be classification error and E2 can be number of false positives) or define satisficing criteria (i.e. use classification error as metric and take into account restraints as memory usage, running time, etc.)
* **Create training, dev and tests set**. See below for details
* **Train multiple models** using training set and checking results against dev set in order to find **errors**


### Training, dev and tests set
- Split available data in 3 sets: **training, development (or cross-validation) and test**.  Development set is fundamental in order to address problems and pick the right model to check against test set. If you had only train and test set, you might end up overfitting on the test set getting bad results with real application data.
- When number of examples is **limited** (i.e < 100,000) a **60/20/20** percentage rule can be applied (60% training, 20% development, 20% test), otherwise when **many data** are available a **98/1/1** percentage rule is preferable (98% training, 1% dev and 1% test).
- All sets should come from **same distribution** as data used in production (i.e. images from the Internet may be very different from images taken by users on smartphones). If it's not possible at least **test and dev sets** must come from same distribution of data used in production (*"Aim your target"* principle) and at least a small part of it should be included in training set; in this case it's highly recommended to divide training set in 2 subsets, **train set and traing-dev set**, in order to have 4 sets (in this case division percentages can be 50/10/20/20 or 97/1/1/1) and check against **data mismatch error**.  Suppose you want to write a mobile app to recognize flowers and you have 10,000 images from source A (shots taken from smartphones just like the ones expected in production) and 1,000,000 images from source B (the Internet). **Do not shuffle A and B between all sets**! Instead divide a shuffled set of 100%A+50%B in two subsets, training and training-dev and the remaining 50% B between dev and test sets.

### Error analysis
In case of same distribution for all sets, set up **different models using your training set** and checks results against your development set in order to **pick one** model to try out with your **test set**. This way you can identify a **bias** problem when error on training set is high or/and a **variance** problem when difference between errors on training and dev set is high.

| Set                       | Case #1      | Case #2            | Case #3                      |
|---------------------|--------------|-------------------|--------------------------|
| Human                 |  0.5%         | 0.5%                 | 0.5%                           |
| Training                |  5%            | 1.0%                 | 5.0%                           |
| Dev                      |  5.5%         | 5.5%                 | 5.5%                           |
| Test                      |  6%            | 6%                    | 10.0%                         |
| Avoidable bias     | 4.5%          | 0.5%                 | 4.5%                           |
| Problem               | High bias   | High variance    | Overfit on dev set       |

In case of different distributions between training/training-dev and dev/test sets,  set up **different models using your training set** and checks results against your training-dev set and your dev set.  This way you can identify a **bias** problem when error on training set is high or/and a **variance** problem when difference between errors on training and training-dev sets is high (since they come from the same distribution) and a **data mismatch error** when difference between errors on training and dev sets is high (if bias and variance are low).

| Set                       | Case #1      | Case #2            | Case #3                      | Case #4                    |
|---------------------|--------------|-------------------|--------------------------|-------------------------|
| Human                 |  0.5%         | 0.5%                 | 0.5%                           | 0.5%                         | 
| Training                |  5%            | 1.0%                 | 1.0%                           | 1.0%                         |
| Training-dev         |  5%            | 5.5%                 | 1.5%                           | 1.5%                         |
| Dev                      |  5.5%         | 6.0%                 | 10.0%                         | 1.5%                         |
| Test                      |  6   %         | 6.5%                 | 10.5%                         | 10.5%                       |
| Avoidable bias     | 4.5%          | 0.5%                 | 0.5%                           | 0.5%                         |
| Problem               | High bias   | High variance    | Data mismatch error   | Overfit on dev set     |

### Error correction
In order to address a **bias** problem, you can try the following:
* Train your model longer
* Add optimization (i.e. Momentum, Adam)
* Add number of hidden layers
* Change number of units in hidden layers
Generally getting more data doesn't help in case of a bias problem

In order to address a **variance** problem you can:
* Use more data for training
* Use **Regularization** (i.e. L2 "weight decay", Dropout, etc.)

### Additional strategies
When starting a new project consider if it's possible **transfer learning**. Specifically if you trained a NN for a specific task (i.e. image recognition) and you need to extend the kinds of images, instead of starting over with training, use weights and biases obtained for layers 1...L-1 (it's called **pre-training**) and random weights for layer L (**fine-tuning**).
Another approach for complex activities which require multiple steps (i.e. in order to recognize the status of a traffic light you might first look for 3 circles, apply filters to check which is lit on, determine status on the basis of its position) is **end-to-end learning** which simply cuts all intermediate steps. In this cas you need a huge amount of data (10 milion or more).

