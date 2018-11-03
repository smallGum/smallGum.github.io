---
layout: post
title: Learn the Neural Networks with Tensorflow (1)
description: "A summary of using Tensorflow to train Neural Networks"
modified: 2018-11-03
tags: [Deep Learning]
image:
  feature: abstract-8.jpg
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

## Introduction

Thanks to Michael Nielsen's book [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) and [*TensorFlow Tutorial and Examples for Beginners with Latest APIs*](https://github.com/aymericdamien/TensorFlow-Examples), I've spent an easier life on learning neural networks. In order to put what I've learned into practice, I decide to repeat the experiments in the book using [**Tensorflow**](https://www.tensorflow.org/), a powerful tool for machine learning.
This blog shows the result of the first experiment, which builds a simple fully-connected neural network to classify MNIST handwritten digits.

## Data Preprocessing

We use the MNIST data provided in [Nielsen's code](https://github.com/mnielsen/neural-networks-and-deep-learning) and his function `load_data()` to load the trainning, validation and test data respectively.


```python
import cPickle
import gzip

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
```

Since we should apply Stochastic Gradient Descent (SGD) method to train our neural network, it's necessary to randomly shuffle our trainning data in order to select batch data in every epoch of training. It's helpful to shuffle trainning data first and then read them in order, which improves cache hits and speeds up learning process.


```python
# randomly shuffle the training data
def random_shuffle(data_features, data_labels):
    indices = np.random.permutation(data_features.shape[0])
    X = data_features[indices]
    Y = data_labels[indices]
    return X, Y
```

Let's read our data and see what we get:


```python
import numpy as np

print 'reading data...'
# read the data from file
training_data, validation_data, test_data = load_data()

# get the trainning data
train_X = training_data[0]
train_Y = training_data[1]
# get the test data
test_X = test_data[0]
test_Y = test_data[1]

print 'training data features: %s, trainning data labels: %s' % (train_X.shape, train_Y.shape)
print 'test data features: %s, test data labels: %s' % (test_X.shape, test_Y.shape)
```

    reading data...
    training data features: (50000, 784), trainning data labels: (50000,)
    test data features: (10000, 784), test data labels: (10000,)


We see that there are 50000 samples in the trainning data and 10000 samples in the test data. Note that we don't use the validation data here, which helps figure out how to set hyper-parameters of our neural network later.

## Neural Network

We use a three-layer neural network with (784, 30, 10) neurons to recognize the digits (0-9, 10 classes):

<figure>
	<img src="/images/learn_tensorflow/network.jpg" alt="neural network structure">
</figure>

We have 10 neurons as output to recognize the digit, meaning that a digit's label should be represented as a 10-dimension vector. For example, if the label of the $i$-th sample $x^{(i)}$ is $4$ , we represent its label as $y^{(i)} = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0)$ ; if the label of the $i$-th sample $x^{(i)}$ is $9$, we represent its label as $y^{(i)} = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)$ , so on and so forth. Therefore, we extend the label of each digit to the shape of (10,). Consequently, the shape of `Y` and `test_Y` would be extended to (50000, 10) and (10000, 10), respectively.


```python
# extend the data from shape(data.shape[0],) to shape(data.shape[0], dim)
def to_vector(data, dim):
    data_vector = np.zeros([data.shape[0], dim], dtype=np.float64)
    for i in range(data.shape[0]):
        index = data[i]
        data_vector[i][index] = 1.0
    return data_vector

trainY_vector = to_vector(train_Y, 10)
testY_vector = to_vector(test_Y, 10)
print 'training data shape: ', trainY_vector.shape
print 'test data shape: ', testY_vector.shape
```

    training data shape:  (50000, 10)
    test data shape:  (10000, 10)


The trainning configuration and hyper-parameters are set as follows:

+ activation function: $\sigma (z) = \frac{1}{1 + e^{-z}}$
+ cost function: $J(\theta) = \frac{1}{2N} \sum_{i = 1}^{N} ||y^{(i)} - h(x^{(i)})||^2$ 
    where $N$ is the total number of training inputs, $y^{(i)}$ is the vector of $i$-th sample's label, $h(x^{(i)})$ is the vector of $i$-th sample's prediction label. $\theta$ contains the weights $w$ and the biases $b$ of neurons.
+ training epochs: $30$
+ mini-batch size: $10$
+ learning rate: $\eta = 3.0$

### Construct the Network with Raw Tensorflow

Next, let's design our neural network step by step using Tensorflow:


```python
import tensorflow as tf

# set the hyper-parameters
epochs = 30
mini_batch_size = 10
learning_rate = 3.0
display_epoch = 5     # display the learning state every 5 epochs

# set the network parameters
input_neurons_num = 784
hidden_1_neurons_num = 30
output_neurons_num = 10

# define the parameter to be fed 
t_X = tf.placeholder(tf.float64, [None, input_neurons_num])
t_Y = tf.placeholder(tf.float64, [None, output_neurons_num])

# define the weights and biases, generated from Gaussian distributions with mean 0 and standard deviation 1
weights = {
    'input_to_h1': tf.Variable(tf.random_normal([input_neurons_num, hidden_1_neurons_num], mean=0.0, stddev=1.0, dtype=tf.float64)),
    'h1_to_output': tf.Variable(tf.random_normal([hidden_1_neurons_num, output_neurons_num], mean=0.0, stddev=1.0, dtype=tf.float64))
}
biases = {
    'h1': tf.Variable(tf.random_normal([hidden_1_neurons_num], mean=0.0, stddev=1.0, dtype=tf.float64)),
    'output': tf.Variable(tf.random_normal([output_neurons_num], mean=0.0, stddev=1.0, dtype=tf.float64))
}

# feedforward process
def predict(x):
    """
    predict the label of trainning data x through the network
    :param x shape[None, input_neurons_num] the trainning data
    :return output_layer shape[None, output_neurons_num] the output vector of training data
    """
    h1_layer = tf.sigmoid(tf.add(tf.matmul(x, weights['input_to_h1']), biases['h1']))
    output_layer = tf.sigmoid(tf.add(tf.matmul(h1_layer, weights['h1_to_output']), biases['output']))
    return output_layer

# construct the model
logit = predict(t_X)

# set the cost function and SGD optimizer
cost_op = tf.reduce_sum(((t_Y - logit) ** 2))  / (2 * mini_batch_size)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost_op)

# evaluate the model with accuracy
correct_num = tf.equal(tf.argmax(logit, 1), tf.argmax(t_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
```

Finally, we train the model and evaluate the result. In order to better observe the learning process, we display the accuracy of the test data every 5 epochs:


```python
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # for each epoch, use mini-batch SGD to train the weights and biases
    for current_epoch in range(0, epochs):
        # randomly shuffle the training data first
        X, Y = random_shuffle(training_data[0], training_data[1])
        # extend the label
        Y_vector = to_vector(Y, 10)
        # run SGD
        for i in range(0, X.shape[0], mini_batch_size):
            batch_x = X[i:(i+mini_batch_size)]
            batch_y = Y_vector[i:(i+mini_batch_size)]
            sess.run(train_op, feed_dict={t_X: batch_x, t_Y: batch_y})
        
        # print the accuracy every 5 epochs
        if (current_epoch+1) % display_epoch == 0 or current_epoch == 0:
            # Calculate accuracy for MNIST test images under current epoch
            print "Epoch %i, Testing Accuracy: %.4f" % (current_epoch+1, sess.run(accuracy, feed_dict={t_X: test_X, t_Y: testY_vector}))
            
    print "Optimization Finished!"
```

    Epoch 1, Testing Accuracy: 0.9053
    Epoch 5, Testing Accuracy: 0.9356
    Epoch 10, Testing Accuracy: 0.9460
    Epoch 15, Testing Accuracy: 0.9494
    Epoch 20, Testing Accuracy: 0.9485
    Epoch 25, Testing Accuracy: 0.9525
    Epoch 30, Testing Accuracy: 0.9510
    Optimization Finished!


The finally accuracy is $95.10\%$, a satisfactory result for such a simple network. Also, the accuracy is close to that in Nielsen's book.

### Construct the Network with `layers` API

The process of constructing our network using raw Tensorflow is quiet tedious and easy to cause bugs, in which we define the weights and biases, mutiply the weights and add the biases manually. Fortunately, Tensorflow provides more powerful API to simplify our work, one of which is the `layers` API. Let's make our life easier!


```python
# set the hyper-parameters
epochs = 30
mini_batch_size = 10
learning_rate = 3.0
display_epoch = 5     # display the learning state every 5 epochs

# set the network parameters
input_neurons_num = 784
hidden_1_neurons_num = 30
output_neurons_num = 10

# define the parameter to be fed 
t_X = tf.placeholder(tf.float64, [None, input_neurons_num])
t_Y = tf.placeholder(tf.float64, [None, output_neurons_num])

# feedforward process
def predict_with_layer(x):
    """
    predict the label of trainning data x through the network using tf.layers
    :param x shape[None, input_neurons_num] the trainning data
    :return output_layer shape[None, output_neurons_num] the output vector of training data
    """
    h1_layer = tf.layers.dense(x, 
                               hidden_1_neurons_num, 
                               activation=tf.sigmoid, 
                               kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float64),
                               bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float64))
    output_layer = tf.layers.dense(h1_layer, 
                                   output_neurons_num,
                                   activation=tf.sigmoid,
                                   kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float64),
                                   bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float64))
    return output_layer

# construct the model
logit = predict_with_layer(t_X)

# set the cost function and SGD optimizer
cost_op = tf.reduce_sum(((t_Y - logit) ** 2))  / (2 * mini_batch_size)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost_op)

# evaluate the model with accuracy
correct_num = tf.equal(tf.argmax(logit, 1), tf.argmax(t_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_num, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
```

Finally, we train our network and evaluate the model with test data. In order to better observe the learning process, we display the accuracy of the test data every 5 epochs:


```python
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # for each epoch, use mini-batch SGD to train the weights and biases
    for current_epoch in range(0, epochs):
        # randomly shuffle the training data first
        X, Y = random_shuffle(training_data[0], training_data[1])
        # extend the label
        Y_vector = to_vector(Y, 10)
        # run SGD
        for i in range(0, X.shape[0], mini_batch_size):
            batch_x = X[i:(i+mini_batch_size)]
            batch_y = Y_vector[i:(i+mini_batch_size)]
            sess.run(train_op, feed_dict={t_X: batch_x, t_Y: batch_y})
        
        # print the accuracy every 5 epochs
        if (current_epoch+1) % display_epoch == 0 or current_epoch == 0:
            # Calculate accuracy for MNIST test images under current epoch
            print "Epoch %i, Testing Accuracy: %.4f" % (current_epoch+1, sess.run(accuracy, feed_dict={t_X: test_X, t_Y: testY_vector}))
            
    print "Optimization Finished!"
```

    Epoch 1, Testing Accuracy: 0.8244
    Epoch 5, Testing Accuracy: 0.9294
    Epoch 10, Testing Accuracy: 0.9373
    Epoch 15, Testing Accuracy: 0.9426
    Epoch 20, Testing Accuracy: 0.9429
    Epoch 25, Testing Accuracy: 0.9463
    Epoch 30, Testing Accuracy: 0.9416
    Optimization Finished!


We see the final accuracy is $94.16\%$, close to the result we first got. The random initilization of weights and biases might affect the accuracy slightly.

## Conclusion

We've repeated the first experiment in Nielsen's book and got a satisfactory result. I'll continue to repeat other experiments using Tensorflow in later blogs, exploring the relative factors that could affect neural network's accuracy, such as the learning rate, the mini-batch size, the regularization, the network structure, etc.
