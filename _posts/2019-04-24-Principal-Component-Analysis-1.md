---
layout: post
title: Principal Component Analysis (I)
subtitle: "Derive the Principal Component Analysis (PCA) algorithm by hand"
date: 2019-04-24 13:20:00
author:     "Maverick"
header-img: "img/post-bg-js-module.jpg"
tags:
    - Machine Learning
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

Principal Component Analysis (PCA) is a famous machine learning algorithm which applies **lossy compression** to raw data for storage or training and reconstructs the original data from the compressed data if necessary. It can be easily derived using the knowledge of [Eigendecomposition](https://smallgum.github.io/2019/04/20/Eigendecomposition/) and [Quadratic Optimization](https://smallgum.github.io/2019/04/23/Quadratic-Optimization/) described in my last two blogs. We would like to introduce the problem solved by PCA and derive this algorithm by hand in this blog.

## Problem Definition

Suppose we have a collection of data samples $$\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, ..., \mathbf{x}^{(n)}\}$$. Each sample $$\mathbf{x}^{(i)} \in \mathbb{R}^d$$ can be treated as a point in $$\mathbb{R}^d$$ space and the $$j$$-th element $$\mathbf{x}^{(i)}_j$$ represents the value of some **feature** of the sample $$\mathbf{x}^{(i)}$$. All data points can be represented as a matrix $$\mathbf{M}=[\mathbf{x}^{(1)}; \mathbf{x}^{(2)}; ...; \mathbf{x}^{(n)}] \in \mathbb{R}^{n \times d}$$. In this case, the matrix $$\mathbf{M}$$ may become very large when the number of features $$d$$ increases, making it difficult and time-consuming to process the raw data. What's worse, irrelevant and redundant features (i.e., features that are highly correlated) in a high dimensional dataset provide little useful information for our analysis and even produce noise that leads to low accuracy. 

To address this problem, dimension of data has to be reduced. One approach is transforming each data point $$\mathbf{x}^{(i)} \in \mathbb{R}^d$$ into a code vector $$\mathbf{c}^{(i)} \in \mathbb{R}^k$$, where $$k < d$$ and thus compress the matrix $$\mathbf{M}$$ into a lower dimensinal matrix $$\mathbf{M}^{'}=[\mathbf{c}^{(1)}; \mathbf{c}^{(2)}; ...; \mathbf{c}^{(n)}] \in \mathbb{R}^{n \times k}$$. Therefore, we need to define an encoding function $$f$$ that produces the code for any input, $$\mathbf{c} = f(\mathbf{x})$$. Besides, in order to reconstruct the original data when necessary, it is important to define a decoding function that produces the reconstructed input given its code, $$\mathbf{x} \approx g(\mathbf{c}) = g(f(\mathbf{x}))$$.

## Derivations

In this section, we derive the PCA algorithm through computing optimal code for decoding and encoding function. We first figure out the decoding function since PCA is mainly defined by the choice of the decoder. And then we utilize the decoding function to decide the encoding function. Finally, we obtain the solution of PCA by optimizing a quadratic function.

### Decoding Function

To simplify the computation, we define the decoder as a matrix $$\mathbf{D} \in \mathbb{R}^{d \times k}$$ and use matrix multiplication to map the code vector back to $$\mathbb{R}^d$$. Hence, the decoding function is defined as follows:

$$
g(\mathbf{c}) = \mathbf{Dc}
$$

In order to keep the compression problem easy, PCA constraints the columns of $$\mathbf{D}$$ to be **orthogonal** to each other. Besides, we constraint all the columns of $$\mathbf{D}$$ to have **unit norm** so that there is a unique solution for the decoding function.

### Encoding Function

For **any** input data point $$\mathbf{x}$$, the optimal corresponding code vector is $$\mathbf{c}^{*}$$ whose deconding $$g(\mathbf{c}^*)$$ well reconstructs the original point $$\mathbf{x}$$. This implies that the difference between $$g(\mathbf{c}^*)$$ and $$\mathbf{x}$$ must be as small as possible. In PCA, we use the square $$L^2$$ norm to find the $$\mathbf{c}^*$$:

$$
\mathbf{c}^* = \mathop{\arg\min}_{\mathbf{c}} \lvert\lvert\mathbf{x} - g(\mathbf{c})\rvert\rvert^2_2
$$

According to the definition of the square $$L^2$$ norm, we obtain:

$$
\begin{align}
\lvert\lvert\mathbf{x} - g(\mathbf{c})\rvert\rvert^2_2 &= (\mathbf{x} - g(\mathbf{c}))^\top(\mathbf{x} - g(\mathbf{c})) \\
&= (\mathbf{x} - \mathbf{Dc})^\top(\mathbf{x} - \mathbf{Dc}) \\
&= (\mathbf{x}^\top - \mathbf{c}^\top\mathbf{D}^\top)(\mathbf{x} - \mathbf{Dc}) \\
&= \mathbf{x}^\top\mathbf{x} - \mathbf{x}^\top\mathbf{Dc} - \mathbf{c}^\top\mathbf{D}^\top\mathbf{x} + \mathbf{c}^\top\mathbf{D}^\top\mathbf{Dc} \\
&= \mathbf{x}^\top\mathbf{x} - 2\mathbf{x}^\top\mathbf{Dc} + \mathbf{c}^\top\mathbf{D}^\top\mathbf{Dc} \\
&= \mathbf{x}^\top\mathbf{x} - 2\mathbf{x}^\top\mathbf{Dc} + \mathbf{c}^\top\mathbf{I}_k\mathbf{c} \\
&= \mathbf{x}^\top\mathbf{x} - 2\mathbf{x}^\top\mathbf{Dc} + \mathbf{c}^\top\mathbf{c}
\end{align}
$$

Note that $$\mathbf{c}^\top\mathbf{D}^\top\mathbf{x} = (\mathbf{c}^\top\mathbf{D}^\top\mathbf{x})^\top = \mathbf{x}^\top\mathbf{Dc}$$ since $$\mathbf{c}^\top\mathbf{D}^\top\mathbf{x}$$ is a scalar, and $$\mathbf{D}^\top\mathbf{D} = \mathbf{I}_k$$ since the columns of $$\mathbf{D}$$ are orthogonal and have unit norm. We omit the first term $$\mathbf{x}^\top\mathbf{x}$$ because it does not depend on $$\mathbf{c}$$:

$$
\mathbf{c}^* = \mathop{\arg\min}_{\mathbf{c}} - 2\mathbf{x}^\top\mathbf{Dc} + \mathbf{c}^\top\mathbf{c}
$$

The optimization problem above is a quadratic function about $\mathbf{c}$, which can be solved by vector calculus:

$$
\begin{align}
\nabla_{\mathbf{c}^{*}}(-2\mathbf{x}^\top\mathbf{Dc}^{*} + \mathbf{c}^{*\top}\mathbf{c}^{*}) &= 0 \\
-2\mathbf{D}^\top\mathbf{x} + 2\mathbf{c}^{*} &= 0 \\
\mathbf{c}^{*} &= \mathbf{D}^\top\mathbf{x}
\end{align}
$$

Therefore, the encoding function is:

$$
f(\mathbf{x}) = \mathbf{D}^\top\mathbf{x}
$$

and the decoding (reconstruction) function is:

$$
r(\mathbf{x}) = g(f(\mathbf{x})) = \mathbf{DD}^\top\mathbf{x}
$$

### Solution of PCA

The solution of PCA is the encoding and decoding matrix $$\mathbf{D}$$. In order to find the optimal matrix $$\mathbf{D}^{*}$$, we minimize the $$L^2$$ distance between inputs and reconstructions for **all data points**:

$$
\begin{align}
\mathbf{D}^{*} &= \mathop{\arg\min}_{\mathbf{D}} \sqrt{\sum_i\sum_j \big(\mathbf{x}^{(i)}_j - r(\mathbf{x}^{(i)})_j\big)^2} \\
&= \mathop{\arg\min}_{\mathbf{D}} \sum_i\sum_j \big(\mathbf{x}^{(i)}_j - r(\mathbf{x}^{(i)})_j\big)^2 \\
&= \mathop{\arg\min}_{\mathbf{D}} \sum_i\sum_j \big(\mathbf{x}^{(i)}_j - (\mathbf{DD}^\top\mathbf{x}^{(i)})_j\big)^2 \\
&= \mathop{\arg\min}_{\mathbf{D}} \lvert\lvert\mathbf{M} - (\mathbf{DD}^\top\mathbf{M}^\top)^\top\rvert\rvert^2_F \\
&= \mathop{\arg\min}_{\mathbf{D}} \lvert\lvert\mathbf{M} - \mathbf{M}\mathbf{DD}^\top\rvert\rvert^2_F \\
& s.t.~~ \mathbf{D}^\top\mathbf{D} = \mathbf{I}_k
\end{align}
$$

where $$\mathbf{M}=[\mathbf{x}^{(1)}; \mathbf{x}^{(2)}; ...; \mathbf{x}^{(n)}] \in \mathbb{R}^{n \times d}$$, $$\lvert\lvert\mathbf{A}\rvert\rvert^2_F$$ denotes the square **Frobenius norm** of the matrix $$\mathbf{A}$$:

$$
\lvert\lvert\mathbf{A}\rvert\rvert^2_F = \big(\sqrt{\sum_{i,j} \lvert\mathbf{A}_{i,j}\rvert^2}\big)^2 = \sum_{i,j} \lvert\mathbf{A}_{i,j}\rvert^2
$$

In order to solve the optimization problem above, we need to use the **Trace Operator**, which gives the sum of all the diagonal entries of a matrix:

$$
\text{Tr}(\mathbf{A}) = \sum_i \mathbf{A}_{i,i}
$$

The trace operator has many useful propertities:

+ The trace operator provides an alternative way of writing the Frobenius norm of a matrix: $$\lvert\lvert\mathbf{A}\rvert\rvert_F = \sqrt{\mathbf{AA}^\top}, ~~~~ \lvert\lvert\mathbf{A}\rvert\rvert_F^2 = \mathbf{AA}^\top$$.
+ The trace operator is invariant to the transpose operator: $$\text{Tr}(\mathbf{A}) = \text{Tr}(\mathbf{A}^\top)$$.
+ The trace of a square matrix composed of many factors is also invariant to moving the last factor into the first position, if the shapes of the corresponding matrices allow the resulting product to be defined: $$\text{Tr}(\mathbf{ABC})=\text{Tr}(\mathbf{CAB})=\text{Tr}(\mathbf{BCA})$$, or more generally, $$\text{Tr}(\prod_i^n \mathbf{F}^{(i)})=\text{Tr}(\mathbf{F}^{(n)}\prod_i^{n-1} \mathbf{F}^{(i)})$$. This can be called the cycle rule.
+ A scalar is its own trace: $$a = \text{Tr}(a)$$.

We now utilize the trace operator to simplify the expression being optimized:

$$
\begin{align}
\mathop{\arg\min}_{\mathbf{D}}\lvert\lvert\mathbf{M} - \mathbf{M}\mathbf{DD}^\top\rvert\rvert^2_F &= \mathop{\arg\min}_{\mathbf{D}}\text{Tr}\big((\mathbf{M} - \mathbf{M}\mathbf{DD}^\top)(\mathbf{M} - \mathbf{M}\mathbf{DD}^\top)^\top\big) \\
&= \mathop{\arg\min}_{\mathbf{D}}\text{Tr}\big((\mathbf{M} - \mathbf{M}\mathbf{DD}^\top)^\top(\mathbf{M} - \mathbf{M}\mathbf{DD}^\top)\big) \\
&= \mathop{\arg\min}_{\mathbf{D}}\text{Tr}\big((\mathbf{M}^\top - \mathbf{DD}^\top\mathbf{M}^\top)(\mathbf{M} - \mathbf{M}\mathbf{DD}^\top)\big) \\
&= \mathop{\arg\min}_{\mathbf{D}}\text{Tr}\big(\mathbf{M}^\top\mathbf{M} - \mathbf{M}^\top\mathbf{MDD}^\top - \mathbf{DD}^\top\mathbf{M}^\top\mathbf{M} + \mathbf{DD}^\top\mathbf{M}^\top\mathbf{MDD}^\top\big) \\
&= \mathop{\arg\min}_{\mathbf{D}} \text{Tr}(\mathbf{M}^\top\mathbf{M}) - \text{Tr}(\mathbf{M}^\top\mathbf{MDD}^\top) - \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{M}) + \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{MDD}^\top) \\
\text{Omit the term Tr}(\mathbf{M}^\top\mathbf{M}) \text{ not involving } \mathbf{D}: \\
&= \mathop{\arg\min}_{\mathbf{D}} - \text{Tr}(\mathbf{M}^\top\mathbf{MDD}^\top) - \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{M}) + \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{MDD}^\top) \\
\text{Cycle the matrices inside a trace operator}: \\
&= \mathop{\arg\min}_{\mathbf{D}} - \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{M}) - \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{M}) + \text{Tr}(\mathbf{M}^\top\mathbf{M}\mathbf{DD}^\top\mathbf{DD}^\top) \\
&= \mathop{\arg\min}_{\mathbf{D}} -2 \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{M}) + \text{Tr}(\mathbf{M}^\top\mathbf{M}\mathbf{D}\mathbf{I}_k\mathbf{D}^\top) \\
&= \mathop{\arg\min}_{\mathbf{D}} -2 \text{Tr}(\mathbf{DD}^\top\mathbf{M}^\top\mathbf{M}) + \text{Tr}(\mathbf{M}^\top\mathbf{M}\mathbf{D}\mathbf{D}^\top) \\
&= \mathop{\arg\min}_{\mathbf{D}} -2 \text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD}) + \text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD}) \\
&= \mathop{\arg\min}_{\mathbf{D}} - \text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD}) \\
&= \mathop{\arg\max}_{\mathbf{D}} \text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD}) \\
& s.t.~~ \mathbf{D}^\top\mathbf{D} = \mathbf{I}_k
\end{align}
$$

Let $$\mathbf{X}=\mathbf{M}^\top\mathbf{M}$$ be the symmetric $$d \times d$$ matrix we obtain. We use $$\mathbf{A}_{i,:}$$ and $$\mathbf{A}_{:,j}$$ to denote the $$i$$-th row and $$j$$-th column of a matrix $$\mathbf{A}$$, respectively. We expand the expression $$\mathbf{D}^\top\mathbf{XD}$$ and get:

$$
\begin{align}
\mathbf{D}^\top\mathbf{XD} &= \begin{bmatrix}
\mathbf{D}_{1,1} & \mathbf{D}_{1,2} & \cdots & \mathbf{D}_{1,k} \\
\mathbf{D}_{2,1} & \mathbf{D}_{2,2} & \cdots & \mathbf{D}_{2,k} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{D}_{d,1} & \mathbf{D}_{d,2} & \cdots & \mathbf{D}_{d,k} \\
\end{bmatrix}^\top \mathbf{X} \begin{bmatrix}
\mathbf{D}_{1,1} & \mathbf{D}_{1,2} & \cdots & \mathbf{D}_{1,k} \\
\mathbf{D}_{2,1} & \mathbf{D}_{2,2} & \cdots & \mathbf{D}_{2,k} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{D}_{d,1} & \mathbf{D}_{d,2} & \cdots & \mathbf{D}_{d,k} \\
\end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{D}_{1,1} & \mathbf{D}_{2,1} & \cdots & \mathbf{D}_{d,1} \\
\mathbf{D}_{1,2} & \mathbf{D}_{2,2} & \cdots & \mathbf{D}_{d,2} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{D}_{1,k} & \mathbf{D}_{2,k} & \cdots & \mathbf{D}_{d,k} \\
\end{bmatrix}  \mathbf{X} \begin{bmatrix}
\mathbf{D}_{1,1} & \mathbf{D}_{1,2} & \cdots & \mathbf{D}_{1,k} \\
\mathbf{D}_{2,1} & \mathbf{D}_{2,2} & \cdots & \mathbf{D}_{2,k} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{D}_{d,1} & \mathbf{D}_{d,2} & \cdots & \mathbf{D}_{d,k} \\
\end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{D}_{1,1} & \mathbf{D}_{2,1} & \cdots & \mathbf{D}_{d,1} \\
\mathbf{D}_{1,2} & \mathbf{D}_{2,2} & \cdots & \mathbf{D}_{d,2} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{D}_{1,k} & \mathbf{D}_{2,k} & \cdots & \mathbf{D}_{d,k} \\
\end{bmatrix} \begin{bmatrix}
\mathbf{X}\mathbf{D}_{:,1} & \mathbf{X}\mathbf{D}_{:,2} & \cdots & \mathbf{X}\mathbf{D}_{:,k}
\end{bmatrix} \\
&= \begin{bmatrix}
\mathbf{D}_{:,1}^\top\mathbf{X}\mathbf{D}_{:,1} & \cdots & \cdots & \cdots \\
\cdots & \mathbf{D}_{:,2}^\top\mathbf{X}\mathbf{D}_{:,2} & \cdots & \cdots \\
\vdots & \vdots & \ddots & \vdots \\
\cdots & \cdots & \cdots & \mathbf{D}_{:,k}^\top\mathbf{X}\mathbf{D}_{:,k} \\
\end{bmatrix} \\
\end{align}
$$

Thus the expression $$\text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD})$$ can be expanded as:

$$
\text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD}) = \sum_{i=1}^{k} \mathbf{D}_{:,i}^\top\mathbf{M}^\top\mathbf{MD}_{:,i}
$$

Hence, $$\text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD})$$ obtain its maximum value when each term $$\mathbf{D}_{:,i}^\top\mathbf{M}^\top\mathbf{MD}_{:,i}$$ is maximized. Let $$\mathbf{d} = \mathbf{D}_{:,i} \in \mathbb{R}^d$$ be the $$i$$-th column of $$\mathbf{D}$$, then $$\mathbf{d}^\top\mathbf{d} = 1$$. According to the [**Rayleigh–Ritz theorem**](https://smallgum.github.io/2019/04/23/Quadratic-Optimization/#rayleighritz-ratio) we introduce in the [Quadratic Optimization](https://smallgum.github.io/2019/04/23/Quadratic-Optimization/) blog, the solution to the following quadratic optimization problem

$$
\mathop{\arg\max}_{\mathbf{d}} \text{Tr}(\mathbf{d}^\top\mathbf{M}^\top\mathbf{Md}) = \mathop{\arg\max}_{\mathbf{d}} \mathbf{d}^\top\mathbf{M}^\top\mathbf{Md} \\
(\text{Tr}(\mathbf{d}^\top\mathbf{M}^\top\mathbf{Md}) = \mathbf{d}^\top\mathbf{M}^\top\mathbf{Md} \text{ since } \mathbf{d}^\top\mathbf{M}^\top\mathbf{Md} \text{ is a scalar.})
$$

is the eigenvector of the symmetric matrix $$\mathbf{M}^\top\mathbf{M}$$ whose corresponding eigenvalue is the maximum eigenvalue. Hence, also following the [corollary of the Rayleigh–Ritz theorem](https://smallgum.github.io/2019/04/23/Quadratic-Optimization/#rayleighritz-ratio), to maximize the value of $$\text{Tr}(\mathbf{D}^\top\mathbf{M}^\top\mathbf{MD})$$ given $$\mathbf{D}^\top\mathbf{D} = \mathbf{I}_k$$, the matrix $$\mathbf{D}$$ is given by the $$k$$ eigenvectors corresponding to the $$k$$ largest eigenvalues of the symmetric matrix $$\mathbf{M}^\top\mathbf{M}$$.

Finally, $$\mathbf{M}^{'} = (\mathbf{D}^\top\mathbf{M})^\top = \mathbf{MD}$$ is the compressed matrix of the raw data matrix.

## Reduce Computational Complexity

We now consider the relationship between the eigenpairs of $$\mathbf{M}^\top \mathbf{M}$$ and $$\mathbf{MM}^\top$$, where $$\mathbf{M} \in \mathbb{R}^{n \times d}$$. When $$n << d$$, which is very common in some applications like face recognition, it will require too much memory to store the matrix $$\mathbf{M}^\top \mathbf{M}$$ and computing its eigenpairs will be a horrendous task. Fortunately, we can find the eigenpairs of $$\mathbf{M}^\top \mathbf{M}$$ through computing the eigenpairs of $$\mathbf{MM}^\top$$. 

Suppose that $$\mathbf{e}$$ is the eigenvector of the matrix $$\mathbf{MM}^\top$$ with corresponding eigenvalue $$\lambda$$, then we have:

$$
\begin{align}
\mathbf{MM}^\top \mathbf{e} &= \lambda \mathbf{e} \\
\text{Multiply both sides by $\mathbf{M}^\top$ on the left:} \\
\mathbf{M}^\top\mathbf{MM}^\top \mathbf{e} &= \mathbf{M}^\top \lambda \mathbf{e} \\
\mathbf{M}^\top\mathbf{M}(\mathbf{M}^\top \mathbf{e}) &= \lambda (\mathbf{M}^\top \mathbf{e})
\end{align}
$$

We can see that when $$\mathbf{M}^\top \mathbf{e} \ne \mathbf{0}$$, $$\mathbf{M}^\top \mathbf{e}$$ is an eigenvector of $$\mathbf{M}^\top\mathbf{M}$$ with corresponding eigenvalue $$\lambda$$. However, when $$\mathbf{M}^\top \mathbf{e} = \mathbf{0}$$, it can not be an eigenvector and we have $$\mathbf{MM}^\top \mathbf{e} = \lambda \mathbf{e} = \mathbf{0}$$, implying $$\lambda = 0$$ since $$\mathbf{e} \ne \mathbf{0}$$ is an eigenvector of $$\mathbf{MM}^\top$$. Hence, we conclude that for a matrix $$\mathbf{M} \in \mathbb{R}^{n \times d}$$, the eigenvalues of $$\mathbf{MM}^\top$$ are the eigenvalues of $$\mathbf{M}^\top\mathbf{M}$$ plus additional 0’s when $$n >> d$$; and the eigenvalues of $$\mathbf{M}^\top\mathbf{M}$$ are the eigenvalues of $$\mathbf{MM}^\top$$ plus additional 0’s when $$n << d$$. 

Finally, when we meet a data matrix $$\mathbf{M} \in \mathbb{R}^{n \times d}$$ where $$n << d$$, we can first compute the eigenpairs of $$\mathbf{MM}^\top$$ and multiply each eigenvector $$\mathbf{e}$$ by $$\mathbf{M}^\top$$ (i.e., $$\mathbf{M}^\top\mathbf{e}$$) to obtain the eigenvectors of $$\mathbf{M}^\top\mathbf{M}$$. In PCA, we only need to select the $$k$$ largest eigenvector of $$\mathbf{MM}^\top$$ for multiplication and thus reduce the computation complexity to $$O(n^3 + kdn^2) = O(n^3 + dn^2)$$, which is much less than $$O(d^3)$$ of computing the eigenpairs of $$\mathbf{M}^\top\mathbf{M}$$. (Remember that $$k$$ is a constant in PCA and it needs $$O(n^3)$$ time to compute the eigenpairs of an $$n \times n$$ symmetric matrix).

## PCA Algorithm

Based on the derivations in the last section, we can now develop the PCA algorithm:

1. Given a set of data sample $$\mathcal{X}=\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, ..., \mathbf{x}^{(n)}\}$$, create the raw data matrix $$\mathbf{M} = [\mathbf{x}^{(1)}; \mathbf{x}^{(2)}; ...; \mathbf{x}^{(n)}]$$.
2. Compute all eigenpairs of the symmetric matrix $$\mathbf{M}^\top\mathbf{M}$$. When the number of columns are much larger than the number of rows, we compute the eigenpairs of $$\mathbf{MM}^\top$$ and use them to find the $$k$$ largest eigenpairs of $$\mathbf{M}^\top\mathbf{M}$$.
3. Use $$k$$ eigenvectors corresponding to $$k$$ largest eigenvalues as columns to form the eigenmatrix $$\mathbf{D}$$.
4. Apply matrix multiplication $$\mathbf{M}^{'}=\mathbf{MD}$$ to compress the raw data.
5. Use $$\mathbf{M}^{r} = \mathbf{M}^{'}\mathbf{D}^\top \approx \mathbf{M}$$ to approximately reconstruct the raw data when necessary.

## Conclusion

In this blog, we define the problem that can be solved by PCA and derive the PCA algorithm by hand. PCA is the core algorithm of many applications such as data compression and using eigenfaces for face recognition. Moreover, from a statistical point of view, PCA actually projects the raw data points onto some directions such that the projections have maximum variance. We'll explain the PCA through specific experiments and in-depth analysis in later blogs.

## Reference

1. [Deep Learning](http://www.deeplearningbook.org/)
2. [CIS 515 Fundamentals of Linear Algebra and Optimization: Chapter 16](http://www.cis.upenn.edu/~cis515/cis515-18-sl14.pdf)
