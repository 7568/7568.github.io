---
layout: blog
text-process: true
background-image: http://7568.github.io/images/2021-11-15-cnn-seq2seqModel/img.png
category: 文本处理
title: 机器翻译 - Seq2Seq with Attention
mathjax: true
tags:
- Seq2Seq
- Attention
- 文本处理
---

[convseq2seq0]:http://7568.github.io/images/2021-11-15-cnn-seq2seqModel/convseq2seq0.png
[convseq2seq1]:http://7568.github.io/images/2021-11-15-cnn-seq2seqModel/convseq2seq1.png

# 简介

在[💝 上一篇blog 💝 ](https://7568.github.io/2021/11/03/rnn-seq2seqModel.html) 中我们讲述了使用rnn来进行自然语言的翻译工作，限于篇幅的原因，我们将会在本blog来讲述使用 cnn 进行自然语言的翻译工作。我们将会在[💝 下一篇 💝 ]() 进行 Transformer 的讲解。

本文我们将会在本blog中实现 [Convolutional Sequence to Sequence Learning ]() 论文中的方法。该方法与我们的之前的方法完全不一样，在之前的方法中我们
使用的都是自然语言处理中常用的循环神经网络rnn，而本文使用的是通常使用在图像处理中任务中的卷积神经网络cnn。不过与通常在图像中使用的cnn不同的是，在图像中cnn的卷积核通常
是带有宽度和高度的，但是在文本处理任务中的cnn卷积核只有长度，没有高度。在[💝 此处 💝 ](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb) 有关于cnn的介绍。

# 准备数据

首先我们还是准备数据，该部分与[ 💝 之前 💝 ](https://7568.github.io/2021/11/03/rnn-seq2seqModel.html) 的内容一致，就不做过多讲解。

# 模型介绍

使用cnn进行文本翻译工作，我们的模型还是分成 encoder 和 decoder 两部分，结构如下图所示。

![convseq2seq0]

## encoder

我们先来看看 encoder 的结构

![convseq2seq1]

我们可以看到在 encoder 中有一个很大的特点就是位置的操作，之前我们的rnn中都没有位置编码，是因为rnn天然就有先后顺序，而cnn没有，
而我们自然语言是有顺序的，相同的单词可能会因为顺序的不一样，组成的句子的意思可能会完全不一样。所以在cnn中需要对输入进行位置编码。

在 encoder 中我们的输入分为6个部分:
1. 将输入进行token化，就是将字符转换成数字。再拼接上位置编码
- 将位置编码与token进行逐点相加，得到带位置属性的token
- 将上一步的结果进行全连接操作
- 将上一步的结果进行卷积操作，得到第一个结果，为卷积层的输出 "conved output"
- 将上一步的结果与第二步的结果进行逐点相加，得到第二个结果，称为 "combined output"

在 rnn 中我们的 encoder 只会有一个结果传到 decoder 中，而在 cnn 中我们有两个结果，分别是"conved output"和"combined output"，都会作为参数传到 decoder 中去。

在上面我们描述的是只有一层 cnn 的网络，如果想有多层，其中一个简单的方法是直接在第4步加上多层网络，本文将介绍一个带残差块的 cnn 网络模型，结构如下图所示：

![convseq2seq1]

在上图中绿色的方块表示gated linear units (GLU)操作，该操作也是跟 GRU 和 LSTM 一样，带有门控单元，是一种带门控的激活函数。

# 代码下载

从[bentrevett / pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq) 中提取出的代码如下：

👉️ 👉️ 👉️ 点击[ 💝 💝 💝 可以直接下载使用 LSTM 结构的 seq2seq 模型的代码](https://7568.github.io/codes/text-process/2021-11-03-seq2seqModel-lstm.py)。将代码中 `is_train = False` 改成 `is_train = True` 就可以训练了，测试的时候再改回来即可。



更多参考资料来自于
- [Towards Data Science - Attention — Seq2Seq Models](https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263)
- [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
-[Jay Alammar Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)



