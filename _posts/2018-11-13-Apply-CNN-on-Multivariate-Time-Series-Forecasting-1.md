---
layout: post
title: Apply CNN on Multivariate Time Series Forecasting (I)
subtitle: "A process of applying CNN on Multivariate Time Series Forecasting"
date: 2018-11-13 16:20:00
author: "Maverick"
header-img: "img/post-bg-android.jpg"
tags: 
    - Deep Learning
catalog: true
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

When talking about Convolutional Neural Network (CNN), we typically think of Computer Vision (CV) or Natural Language Processing (NLP). CNN was responsible for major breakthroughs in both Image Classification and Text Mining.

More recently, some researchers also start to apply CNNs on Multivariate Time Series Forecasting and get results better than traditional Autoregression model, such as [Vector Autoregression (VAR)](https://en.wikipedia.org/wiki/Vector_autoregression). In this blog I'm going to explain how to apply CNNs on Multivariate Time Series and some related concepts.

## Goal

Time Series Forecasting focuses on predicting the future values of variables we considered given their past, which amounts to expressing the expectation of future values as a function of the past observations:

$$
\mathbb{E}[X_{t+d}|X_{t-1}, X_{t-2}, ..., X_{0}] = f(X_{t-1}, X_{t-2}, ..., X_{0}) ~~~~d = 0, 1, 2, ...
$$

Where $X_{i} = (x_{i}^1, x_{i}^2, ..., x_{i}^n)$ is the vector of all concerned variables' observations at time $i$. For example, we have following time series:

$$
\text{Table 1: Raw Time Series Data}\\
\\
\begin{array}{c|ccccc}
time & x^1 & x^2 & x^3 & x^4 & x^5 \\
\hline
t & ? & ? & ? & ? & ? \\
t-1 & 0.412 & -0.588 & 0.618 & 0.673 & 0.661 \\
t-2 & 0.399 & 1.399 & 0.598 & -0.388 & 0.416 \\
t-3 & 0.368 & 1.368 & 0.552 & 0.623 & 0.247 \\
t-4 & 0.338 & -0.662 & 0.169 & 0.266 & 0.413 \\
t-5 & 0.297 & -0.703 & 0.148 & 0.570 & 0.140 \\
t-6 & 0.285 & 1.368 & 0.552 & 0.623 & 0.247
\end{array}
$$

We consider $$X_{t-1}=({x^1_{t-1}, x^2_{t-1}, x^3_{t-1}, x^4_{t-1}, x^5_{t-1}})=({0.412, -0.588, 0.618, 0.673, 0.661}), X_{t-2}=({0.399, 1.399, 0.598, -0.388, 0.416})$$, 
so on and so forth. Our goal is to figure out the values of 
$$\mathbb{E}[X_{t}|X_{t-1}, X_{t-2}, ..., X_{t-6}]$$, 
that is, the values of $$?$$ in the above table. 

## CNN Model

We design following CNN architecture:

```
Input -> Conv -> LeakyReLU -> Pool -> Conv -> LeakyReLU -> Dense
```

Next, we will go through the training process of our CNN step by step, using the example time series shown in $\text{Table 1}$. We'd like to figure out what CNN is doing with those data.

### Part 1

$\text{Figure 1}$ depicts the first part `Input -> Conv -> LeakyReLU -> Pool` of our CNN:

$$
\text{Figure 1: Part 1 of CNN}
$$

<figure>
	<img src="/images/time_series/CNN1.jpg" alt="Part 1 of CNN">
</figure>

#### Input Layer & Dropout

For the Input Layer, we just input our data shown in $\text{Table 1}$. Before moving forward to Convolutional Layer, we apply `Dropout` with `dropout_rate=0.5` on the input data. Assume that we finally get the following new dataset:

(For how to apply `Dropout` in CNN, please refer to **Section 3.2** in [*Towards Dropout Training for Convolutional Neural Networks*](https://arxiv.org/pdf/1512.00242.pdf))

$$
\text{Table 2: Time Series Data after Dropout}\\
\\
\begin{array}{c|ccccc}
time & x^1 & x^2 & x^3 & x^4 & x^5 \\
\hline
t-1 & 0.412 & -0.588 & 0 & 0.673 & 0.661 \\
t-2 & 0 & 1.399 & 0.598 & 0 & 0 \\
t-3 & 0.368 & 0 & 0.552 & 0.623 & 0.247 \\
t-4 & 0 & -0.662 & 0.169 & 0 & 0.413 \\
t-5 & 0.297 & 0 & 0.148 & 0 & 0.140 \\
t-6 & 0.285 & 0 & 0.552 & 0.623 & 0
\end{array}
$$

#### Convolutional Layer

For the Convolutional Layer, the convolution operation is quite similar to [that in NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/). We design 2 kernels of `shape(1, 5)` (the first and second kernel in $\text{Figure 1}$) and 2 kernels of `shape(3, 5)` (the third and fourth kernel in $\text{Figure 1}$). For simplicity, we initilize all the kernels and their biases randomly, using the *Numpy* `np.random.randn` function to generate Gaussian distributions with mean $0$ and standard deviation $1$:

$$
\begin{align}
& K_{1} = (-0.789, -0.288, -0.533, 1.287, 0.693), ~~~~ Bias_{1} = 0.697 \\
& K_{2} = (0.388, -0.717, 1.017, 0.354, -0.474), ~~~~ Bias_{2} = -0.858 \\
& K_{3} = {
\left[ \begin{array}{ccccc}
0.276 & 1.705 & 1.218 & -1.133 & -0.152 \\
0.901 & -1.108 & -2.224 & -1.138 & 1.989 \\
1.177 & -0.585 & 0.134 & -0.723 & -0.301
\end{array} 
\right ]} ~~~~ Bias_{3} = -0.663 \\
& K_{4} = {
\left[ \begin{array}{ccccc}
0.727 & -0.124 & 0.307 & 0.900 & 1.004 \\
-1.417 & -0.685 & -0.250 & 1.247 & -0.869 \\
0.423 & -1.108 & 1.163 & -0.512 & -0.839
\end{array} 
\right ]} ~~~~ Bias_{4} = 0.293
\end{align}
$$

We just multiply the kernels of `shape(1, 5)` with $X_{i}$ in each row of $\text{Table 2}$ with **element wise product**, and add the biases to create elements of filtered vector (the $V_{K_{1}}$ ~ $V_{K_{4}}$ vectors in $\text{Figure 1}$):

$$
\begin{equation}
V_{K_{1}} = {
\left[ \begin{array}{c}
X_{t-1} \odot K_{1} + Bias_{1} \\
X_{t-2} \odot K_{1} + Bias_{1} \\
X_{t-3} \odot K_{1} + Bias_{1} \\
X_{t-4} \odot K_{1} + Bias_{1} \\
X_{t-5} \odot K_{1} + Bias_{1} \\
X_{t-6} \odot K_{1} + Bias_{1}
\end{array} 
\right ]} \\ = {
\left[ \begin{array}{c}
0.412 \times (-0.789)+(-0.588) \times (-0.288)+0 \times (-0.533)+0.673 \times 1.287+0.661 \times 0.693+0.697 \\
0 \times (-0.789)+1.399 \times (-0.288)+0.598 \times (-0.533)+0 \times 1.287+0 \times 0.693+0.697 \\
0.368 \times (-0.789)+0 \times (-0.288)+0.552 \times (-0.533)+0.623 \times 1.287+0.247 \times 0.693+0.697 \\
0 \times (-0.789)+(-0.662) \times (-0.288)+0.169 \times (-0.533)+0 \times 1.287+0.413 \times 0.693+0.697 \\
0.297 \times (-0.789)+0 \times (-0.288)+0.148 \times (-0.533)+0 \times 1.287+0.140 \times 0.693+0.697 \\
0.285 \times (-0.789)+0 \times (-0.288)+0.552 \times (-0.533)+0.623 \times 1.287+0 \times 0.693+0.697 \\
\end{array} 
\right ]} \\ = {
\left[ \begin{array}{c}
1.8655 \\
-0.025 \\
1.085 \\
1.084 \\
0.481 \\
0.980 \\
\end{array} 
\right ]}
\end{equation}
$$

Similiarly, we can obtain $V_{K_{2}}$ by convolving it and the second kernel: 

$$
\begin{equation}
V_{K_{2}} = {
\left[ \begin{array}{c}
X_{t-1} \odot K_{2} + Bias_{2} \\
X_{t-2} \odot K_{2} + Bias_{2} \\
X_{t-3} \odot K_{2} + Bias_{2} \\
X_{t-4} \odot K_{2} + Bias_{2} \\
X_{t-5} \odot K_{2} + Bias_{2} \\
X_{t-6} \odot K_{2} + Bias_{2}
\end{array} 
\right ]} = {
\left[ \begin{array}{c}
0.506 \\
-0.395 \\
0.808 \\
0.451 \\
0.199 \\
0.893
\end{array}
\right ]}
\end{equation}
$$

Particularly, for the third and fourth kernel of `shape(3, 5)`, we use the ["SAME" pattern in *Keras*](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t) to apply $0$-padding on both end of the data in order to make sure the $V_{K_{3}}$ and $V_{K_{4}}$ we get have the same `shape` of $V_{K_{1}}$ and $V_{K_{2}}$. Unlike the first and second kernel, we should apply element wise product between **every three rows** (i.e.,$X_{i}, X_{i+1}, X_{i+2}$) and $K_{3}$ to obtain $V_{K_{3}}$:

$$
\begin{equation}
V_{K_{3}} = {
\left[ \begin{array}{c}
\sum{(\vec 0 \odot K_{3}^{(0)} + X_{t-1} \odot K_{3}^{(1)} + X_{t-2} \odot K_{3}^{(2)})} + Bias_{3} \\
\sum{(X_{t-1} \odot K_{3}^{(0)} + X_{t-2} \odot K_{3}^{(1)} + X_{t-3} \odot K_{3}^{(2)})} + Bias_{3} \\
\sum{(X_{t-2} \odot K_{3}^{(0)} + X_{t-3} \odot K_{3}^{(1)} + X_{t-4} \odot K_{3}^{(2)})} + Bias_{3} \\
\sum{(X_{t-3} \odot K_{3}^{(0)} + X_{t-4} \odot K_{3}^{(1)} + X_{t-5} \odot K_{3}^{(2)})} + Bias_{3} \\
\sum{(X_{t-4} \odot K_{3}^{(0)} + X_{t-5} \odot K_{3}^{(1)} + X_{t-6} \odot K_{3}^{(2)})} + Bias_{3} \\
\sum{(X_{t-5} \odot K_{3}^{(0)} + X_{t-6} \odot K_{3}^{(1)} + \vec 0 \odot K_{3}^{(2)})} + Bias_{3} \\
\end{array}
\right ]} = {
\left[ \begin{array}{c}
0.833 \\
-4.650 \\
2.285 \\
1.537 \\
-0.810 \\
-1.439
\end{array}
\right ]}
\end{equation}
$$

Where $K_{3}^{(i)}$ denotes the $i$-th row of $K_{3}$ and $\sum{(V)}$ denotes the sum of all the elements in vector $V$. Similiarly, we get $V_{K_4}$:

$$
\begin{equation}
V_{K_{4}} = {
\left[ \begin{array}{c}
\sum{(\vec 0 \odot K_{4}^{(0)} + X_{t-1} \odot K_{4}^{(1)} + X_{t-2} \odot K_{4}^{(2)})} + Bias_{4} \\
\sum{(X_{t-1} \odot K_{4}^{(0)} + X_{t-2} \odot K_{4}^{(1)} + X_{t-3} \odot K_{4}^{(2)})} + Bias_{4} \\
\sum{(X_{t-2} \odot K_{4}^{(0)} + X_{t-3} \odot K_{4}^{(1)} + X_{t-4} \odot K_{4}^{(2)})} + Bias_{4} \\
\sum{(X_{t-3} \odot K_{4}^{(0)} + X_{t-4} \odot K_{4}^{(1)} + X_{t-5} \odot K_{4}^{(2)})} + Bias_{4} \\
\sum{(X_{t-4} \odot K_{4}^{(0)} + X_{t-5} \odot K_{4}^{(1)} + X_{t-6} \odot K_{4}^{(2)})} + Bias_{4} \\
\sum{(X_{t-5} \odot K_{4}^{(0)} + X_{t-6} \odot K_{4}^{(1)} + \vec 0 \odot K_{4}^{(2)})} + Bias_{4} \\
\end{array}
\right ]} = {
\left[ \begin{array}{c}
-0.771 \\
0.805 \\
0.496 \\
1.478 \\
0.413 \\
0.637
\end{array}
\right ]}
\end{equation}
$$

Finally, concatenating all $V_{K_{i}}$ vectors generates the output of the Convolutional Layer:

$$
\text{conv_out} = {
\left[ \begin{array}{cccc}
V_{K_{1}} & V_{K_{2}} & V_{K_{3}} & V_{K_{4}}
\end{array}
\right ]} = {
\left[ \begin{array}{cccc}
1.866 & 0.506 & 0.833 & -0.771 \\
-0.025 & -0.395 & -4.650 & 0.805 \\
1.085 & 0.808 & 2.285 & 0.496 \\
1.084 & 0.451 & 1.537 & 1.478 \\
0.481 & 0.199 & -0.810 & 0.413 \\
0.980 & 0.893 & -1.439 & 0.637
\end{array}
\right ]}
$$

#### Batch Normalization

Next, we move to Batch Normalization. Since we only have one data example rather than a "batch" of training dataset, we just use the mean and variance of each $V_{K_{i}}$ to normalize them and initilize $\gamma, \beta$ from Gaussian distributions with mean $0$ and standard deviation $1$:

$$
\gamma = 1.559, \beta = -0.450, \epsilon = 0.01 \\
\text{E}[V_{K_{1}}] = 0.912, \text{E}[V_{K_{2}}] = 0.410, \text{E}[V_{K_{3}}] = -0.374, \text{E}[V_{K_{4}}] = 0.510 \\
\text{Var}[V_{K_{1}}] = 0.340, \text{Var}[V_{K_{2}}] = 0.183, \text{Var}[V_{K_{3}}] = 5.298, \text{Var}[V_{K_{4}}] = 0.448 \\
\text{BN_out} = {
\left[ \begin{array}{cccc}
\gamma \frac{V_{K_{1}} - \text{E}[V_{K_{1}}]}{\sqrt{\text{Var}[V_{K_{1}}] + \epsilon}} + \beta & \gamma \frac{V_{K_{2}} - \text{E}[V_{K_{2}}]}{\sqrt{\text{Var}[V_{K_{2}}] + \epsilon}} + \beta & \gamma \frac{V_{K_{3}} - \text{E}[V_{K_{3}}]}{\sqrt{\text{Var}[V_{K_{3}}] + \epsilon}} + \beta & \gamma \frac{V_{K_{4}} - \text{E}[V_{K_{4}}]}{\sqrt{\text{Var}[V_{K_{4}}] + \epsilon}} + \beta
\end{array}
\right ]} \\ = {
\left[ \begin{array}{cccc}
2.0640 &  -0.1093  &  0.3667 &  -3.4010 \\
-2.9192 &  -3.3067 &  -3.3435 &   0.2296 \\
0.0059  &  0.9624  &  1.3493 &  -0.4823 \\
0.0033 &  -0.3045  &  0.8431 &   1.7799 \\
-1.5858 &  -1.1988 &  -0.7450 &  -0.6735 \\
-0.2708 &   1.2640 &  -1.1707 &  -0.1574 \\
\end{array}
\right ]}
$$

(For how to apply `Batch Normalization` in CNN, please refer to [*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://arxiv.org/pdf/1502.03167v3.pdf) and [*Batch Normalization in Convolutional Neural Network*](https://stackoverflow.com/questions/38553927/batch-normalization-in-convolutional-neural-network))

#### Leaky ReLU Activation

Next, we should apply the Leaky ReLU function to each elements in $\text{BN_out}$:

$$
LeakyReLU(x)={\begin{cases}x&{\mbox{if }}x>0\\0.01x&{\mbox{otherwise}}\end{cases}}\\
\text{Act_out} = LeakyReLU(\text{BN_out}) = {
\left[ \begin{array}{cccc}
2.0640 &  -0.001  &  0.3667 &  -0.034 \\
-0.029 &  -0.033 &  -0.033 &   0.2296 \\
0.0059  &  0.9624  &  1.3493 &  -0.005 \\
0.0033 &  -0.003  &  0.8431 &   1.7799 \\
-0.016 &  -0.012 &  -0.007 &  -0.007 \\
-0.000 &   1.2640 &  -0.012 &  -0.002 \\
\end{array}
\right ]}
$$

#### Max Pooling

At the end of the first part, we apply Max Pooling on each $V_{K_{i}}$ of $\text{BN_out}$. Suppose the `pooling_size=2`, we get:

$$
\text{Pooling_out} = {
\left[ \begin{array}{cccc}
2.0640  &  -0.001  &  0.3667 &  0.2296 \\
0.0059  &  0.9624  &  1.3493 &  1.7799 \\
-0.000  &  1.2640  & -0.007 &  -0.002 \\
\end{array}
\right ]}
$$

### Part 2

$\text{Figure 2}$ depicts the second part `Pool -> Conv -> LeakyReLU -> Dense` of our CNN:

$$
\text{Figure 2: Part 2 of CNN}
$$

<figure>
	<img src="/images/time_series/CNN2.jpg" alt="Part 2 of CNN">
</figure>

All the operations in Part 2 are similar to Part 1. After the Dense Layer, we'll eventually get the $\mathbb{E}[X_{t}]$ which will be used to calculate the loss function during training.

## Conclusion

We just have gone through the whole training process of one time series sample in CNN, illustrating the concepts of `Dropout`, `Convolution`, `Batch Normalization` and `Max Pooling`. I hope this process could be helpful to understand how the CNN works with Multivariate Time Series Forecasting. In the next blog, I'll train a CNN model for Multivariate Time Series Forecasting using **Tensorflow**.
