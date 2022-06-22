---
layout: blog
time-series-process: true
mathjax: true
date:   2021-04-26
background-image: https://7568.github.io/images/2021-04-26-machine-learning-for-parametric-PDEs-and-financial-derivative-pricing/img.png
title:  Monte Carlo Simulations With Deep Leaning
category: time series 处理
tags:
- tabular data
- time series
---

[tabnet-architecture]:https://7568.github.io/images/2021-11-26-tabnet/tabnet-architecture.png

# 简介

本文介绍的是论文[Self-learning Monte Carlo with Deep Neural Networks](https://arxiv.org/pdf/1801.01127.pdf) 
中的内容，这篇论文讲述的是利用蒙特卡罗模拟和自监督学习的方式来训练一个神经网络，这样就可以用神经网络来代替蒙特卡罗模拟，从而可以大大缩减模拟的时间。

# 正文

Self-learning Monte Carlo (SLMC) 是一个用来加速 蒙特卡罗(Monte Carlo （MC）) 模拟的常规算法。通过一个能在指定配置的空间中提供全局移动方法的有效模型，使得 SLMC 
在多个系统中展示出有效性。在本文中，我们的研究显示，深度神经网络能够很自然的整合到 SLMC 中，而且不需要任何先验知识，就能够学到一个准确度高而且有效性好
的模型。我们还展示了在有量化杂志的模型中，我们减少了本地更新的复杂度，从 $$\mathcal {O}(\beta_2)$$ 降到了 $$\mathcal {O}(\beta ln \beta)$$ ，
这会大大的提高执行的效率。

作为一个无偏的方法，蒙特卡罗 (Monte Carlo （MC）)模拟在理解凝聚态系统中发挥着重要的作用。在过去的几十年中，虽然取得了巨大的成功，但是仍然有一些有趣的系统实实在在的超过了普通的MC方法，
又由于这些方法在本地更新的时候有很强的自相关性或者是对于单个的本地更新需要很大的计算消耗。于是随着深度学习技术在物理学中的发展， 一些人提出了 Self-learning Monte Carlo (SLMC) 方法来解决这些问题。
首先提出来的是基于统计机制的模型，然后就是经典的自旋费米子模型，再然后就是行列式量子蒙特卡罗（DQMC），最后是连续时间量子蒙特卡罗和混合蒙特卡罗。最近，通过在DQMC模拟中设置一个
新的系统大小的记录，从而能够帮助理解巡回量子的决策点。

SLMC的设计思想是先学再赚，所以 SLMC 的核心成份是一个有效模型，该模型经过训练之后能够动态的将原始模型整合进来。SLMC 的优点又两方面：一是这个有效的模型模拟的速度很快，它能够使机器在原模型上
提出一个全局的移动，从而来加速MC的模拟。二是该有效模型能够直接的揭示底层的物理意义，例如在 double-exchange 模型中的 RKKY 交互，和本地 spin-spin imaginary-time 的相关性。
我们注意到在以前已经有很多的工作来结合有效潜力方法，或者提出不同的全局移动方法来提高蒙特卡罗模拟的有效性。

SLMC 的有效性是基于有效模型的准确性，它通常是基于人类对原系统的理解上的创新。当将SLMC进一步扩展到那些有效模型有效性的准确率难以描写的复杂系统中的时候，我们使用深度神经网络作为SLMC中的有效模型。



