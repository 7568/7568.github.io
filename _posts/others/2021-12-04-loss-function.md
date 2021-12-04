---
layout: blog
others: true
istop: true
mathjax: true
title: "机器学习中的损失函数"
background-image: http://7568.github.io/images/2021-12-04-loss-function/img.png
date:  2021-12-04
category: 其他
tags:
- loss function
- deep learning
---

[l1-l2-smoothl1]:http://7568.github.io/images/2021-12-04-loss-function/l1-l2-smoothl1.png
[l1-loss-2]:http://7568.github.io/images/2021-12-04-loss-function/l1-loss-2.png

# 引言

在机器学习中，有各种个样的损失函数，不同的损失函数对于不同的任务通常会有不同的效果，而且在有些任务中，大家会使用多个损失函数联合起来，一起训练网络。
本文将搜集一些以前人们设计出来的损失函数。

## L1 Loss

首先我们计算输入$$x$$和目标$$y$$中每个元素的绝对误差，方法如下：

$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = \left| x_n - y_n \right|$$

然后我们对绝对误差可以选择求和或者取平均，这样就得到了L1 Loss。

$$\ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}$$

L1 Loss 通常比较适用于回归任务。L1 Loss 的缺点是在0的位置不可导。L1 Loss 的函数图会与 SmoothL1 Loss 和 L2 Loss 放在 L2 Loss 部分一起做对比。

## SmoothL1 Loss

SmoothL1 Loss 是在 L1 Loss 进行的修改，使得 L1 Loss 的在 0 的位置变得可导。
首先我们还是要的到输入$$x$$和目标$$y$$中每个元素的误差。

$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top$$
  
计算误差的方法如下：

$$l_{i} =
        \begin{cases}
        0.5 (x_i - y_i)^2/beta, & \text{if } |x_i - y_i| < beta \\
        |x_i - y_i| - 0.5 * beta, & \text{otherwise }
        \end{cases}$$
  
然后我们对误差可以选择求和或者取平均，这样就得到了L1 Loss。

$$\ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}$$
  
SmoothL1 Loss 是 L1 Loss 的改进版本，通常在使用 L1 Loss 的地方都能使用 SmoothL1 Loss ，对于不同的任务，他们俩应该是会有稍微的不同。
SmoothL1 Loss 的函数图会与 L1 Loss 和 L2 Loss 放在 L2 Loss 部分一起做对比。

## L2 Loss（MSELoss）

L2 Loss 与 L1 Loss 其实是很相近的，不同点是 L1 Loss 计算的是两个元素的绝对距离，而 L2 Loss 计算的是两个元素的均方误差。
计算方式如下：

$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = \left( x_n - y_n \right)^2$$
  
然后我们对均方误差可以选择求和或者取平均，这样就得到了L2 Loss。

$$\ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}$$
  
通常 L2 Loss ，L1 Loss 和 SmoothL1 Loss，他们的适用范围是一样的。我们在实际的深度学习任务中，可以都尝试以下这三个方法，比较他们之间的差别，从而可以得到当前任务合适的 Loss。

L2 Loss ，L1 Loss 和 SmoothL1 Loss 他们的对比图如下：

![l1-l2-smoothl1]

## CrossEntropy Loss

CrossEntropy Loss 也叫交叉熵损失函数。该损失函数通常用于深度学习中的分类任务。比如分类任务的类别为$$C$$，那么我们可以将类别用$$[0,1,...,C-1]$$来标志类别。
CrossEntropy Loss 函数可表示成如下：

$$\text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)$$

## NLL Loss

## PoissonNLL Loss

## KLDiv Loss

## BCE Loss

## BCEWithLogits Loss

## HingeEmbedding Loss

## MultiLabelMargin Loss

## SoftMargin Loss

## MultiLabelSoftMargin Loss

## CosineEmbedding Loss

## MarginRanking Loss

## MultiMargin Loss

## TripletMargin Loss

## CTC Loss





