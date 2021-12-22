---
layout: blog
others: true
istop: true
mathjax: true
title: "网络剪枝分析"
background-image: https://7568.github.io/images/2021-12-22-pruning-for-neural-network/img.png
date:  2021-12-22
category: 其他
tags:
- deep compression
- deep learning
---

[three-stage-compression-pipeline]:https://7568.github.io/images/2021-12-14-deep-compression/three-stage-compression-pipeline.png
[matrix-sparsity-relative-index]:https://7568.github.io/images/2021-12-14-deep-compression/matrix-sparsity-relative-index.png
[weight-sharing]:https://7568.github.io/images/2021-12-14-deep-compression/weight-sharing.png

# 简介

[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) 是 [Song Han](https://songhan.mit.edu/) 在Stanford大学的时候发表的一篇关于网络如何减少参数的论文。

# Abstract

神经网络对计算和存储的要求都很高，使得他们一般很难不是到嵌入式系统中。而且，常规的网络在训练之前他的结构都是固定的，所以导致的结果就是训练的时候不能实时的优化结构。为了处理这一限制，我们提出了一种方法来
大幅度减少神经网络对存储和计算的需求，同时还不会影响网络的精确度。我们的方法使用三步来剪掉网络中多余的连接。首先，我们将网络训练，使得它能学习到哪些连接是重要的。第二步，我们剪掉那些不重要的连接。
最后，我们重新训练我们的网络来微调那些保存下来的连接。在ImageNet数据集上，我们的方法能将AlexNet的参数数量减少49倍，从61 million 减少到 6.7 million，而且还没有精度的损失。同样的实验用在VGG-16上，参数减少了13倍，
从138 million 较少到了 10.3 million，同样没有损失精度。

# Introduction

