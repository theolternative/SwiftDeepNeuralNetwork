# How To
This file is intended as a super synopsis of all basic concepts in order to develop Machine Learning projects. Specifically it's a simplicistic and condensed summary (did I say it's very basic?) of topics treated in [Deep Learning Specialization Course](https://www.coursera.org/specializations/deep-learning).

## Structuring Machine Learning Projects

### Definitions
**Accuracy** is the ratio between correct vs total number of predictions
**Error** is 1.0-Accuracy
**Bayes error** is the lowest possible prediction error. Nor humans nor AI can achieve lower values.
**Human error** is the mean human prediction error. Human error >= Bayes error
**Bias** is the error on training set
**Avoidable bias** is the difference between error on training set and Bayes error (or human error if more meaningful)
**Variance** is the difference between error on test set and error on training set
**Data mismatch error** is the error due to data coming from different distributions

### Deep Learning Cycle
*  **Pick a metric** (i.e. classification error). In case of multiple criteria, create a unique evaluation (i.e. harmonic mean = 2 / (1/E1+1/E2) where E1 can be classification error and E2 can be number of false positives) or define satisficing criteria (i.e. use classification error as metric and take into account restraints as memory usage, running time, etc.)
* **Create training, dev and tests set**. See below for details
* **Train multiple models** using training set and checking results against dev set in order to find **errors**


### Training, dev and tests set
- Split available train data in 3 sets: **training, development (or cross-validation) and test**. 
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
