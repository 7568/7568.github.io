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

[figure_1]:https://7568.github.io/images/2021-12-22-pruning-for-neural-network/figure_1.png

# 简介

[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) 是 [Song Han](https://songhan.mit.edu/) 在Stanford大学的时候发表的一篇关于网络如何减少参数的论文。

# Abstract

神经网络对计算和存储的要求都很高，使得他们一般很难不是到嵌入式系统中。而且，常规的网络在训练之前他的结构都是固定的，所以导致的结果就是训练的时候不能实时的优化结构。为了处理这一限制，我们提出了一种方法来
大幅度减少神经网络对存储和计算的需求，同时还不会影响网络的精确度。我们的方法使用三步来剪掉网络中多余的连接。首先，我们将网络训练，使得它能学习到哪些连接是重要的。第二步，我们剪掉那些不重要的连接。
最后，我们重新训练我们的网络来微调那些保存下来的连接。在ImageNet数据集上，我们的方法能将AlexNet的参数数量减少49倍，从61 million 减少到 6.7 million，而且还没有精度的损失。同样的实验用在VGG-16上，参数减少了13倍，
从138 million 较少到了 10.3 million，同样没有损失精度。

# Introduction

在计算机视觉，语音识别，自然语言处理中，神经网络已经无处不在。将卷积神经网络应用到计算机视觉中，已经发展了很长时间了。在1998年Lecun等人设计了一个用来识别手写字的网络模型LeNet-5，该模型的参数少于1M，
2012年的时候 Krizhevsky 设计了一个网络模型，赢得了当年的 ImageNet 数据集分类任务的比赛，该模型的参数只有60M，等。

虽然这些大规模的神经网络很强大，但是他们需要相当大的保存，缓存，和计算资源。对于嵌入式的手机而言，这些必须的资源是无法满足的。图1显示了在 45nm CMOS 处理器上的这些基本的算法在
耗能和存储上的消耗。从这些数据中我们可以看到，每一层上的能量的消耗主要在内存的访问上，范围从32 bits的 on-chip SRAM中耗能5pJ到64 bits的off-chip DRAM中耗能640pJ。
大型的网络并不适合在 on-chip 上存储，因此需要更昂贵的 DRAM 来存储。运行连接数有1 billion的一个网络，例如，在20Hz的频率下，只是对于 DRAM 就需要 (20Hz)(1G)(640pJ) = 12.8W 
的能耗，已经超出了普通手机的能量范围。我们对网络进行剪枝的目的就是为了减少运行大规模网络所需要的能量消耗，可以使他们能实时的运行在手机上。通过剪枝后的模型，使得整合了DNN的手机程序依然能够方便的存储和转换到手机上。
![figure_1]


