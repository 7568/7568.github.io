---
layout: blog
others: true
istop: true
mathjax: true
title: "深度压缩论文学习"
background-image: http://7568.github.io/images/2021-12-14-deep-compression/img.png
date:  2021-12-14
category: 其他
tags:
- deep compression
- deep learning
---

# 简介

[DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING]() 是 Song Han 在 2016 年发表在 cs.cv 上的一篇论文。
该论文主要讲述的是通过一种新方法是神经网络的参数大幅减少，同时网络的性能基本上不变，例如将 AlexNet 的参数从 240MB 减少到6.9M，这时十分惊人的效果。接下来我就通过论文和代码来分析一下论文。

#摘要

神经网络通常需要很大的计算量和存储量，从而使得它很难在资源受限的嵌入式系统上部署。为了处理这种限制，我们提出了一种深度压缩的方法："deep compression"，该方法分成三个步骤来进行压缩，分别是剪枝、
量化处理、 Huffman 编码。这三个步骤共同作用，使得网络所需要的存储能缩小到的35倍或者49倍，而不影响网络的精度。我们方法的第一步就是通过学习来给网络剪枝，只留下重要的连接层。接下来，我们我们量化分析权重参数，使得网络能有
更多的权重共享。最后我们使用 Huffman 编码 。在经过了前两步之后，我们重新训练网络，来微调剩下的连接层，和量化之后的中心点。剪枝通常能将网络连接的层数减少9到13倍，然后量化分析能将每层连接的表示字节从32变到5