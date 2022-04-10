---
layout: blog
others: true
istop: true
title: "action recognition"
background-image: http://7568.github.io/images/2022-04-09-synthetic-humans-for-action-recognition-from-unseen-viewpoints.md/img.png
date:  2022-04-09
category: 视频动作识别
tags:
- video
- action recognition
- Synthetic data
---

[figure_1]:https://7568.github.io/images/2022-03-23-my-deep-compression-note/figure_1.png



[Synthetic Humans for Action Recognition from Unseen Viewpoints](https://arxiv.org/abs/1912.04070)
摘要：本文想通过利用合成的数据来提升视频人类动作的识别率。基于此想法，作者设计了一套合成数据的生成方法，生成了一个新的数据集，然后通过在该数据集上进行训练，再分别在NTU RGB+D 和 UESTC 数据集上做
微调，最后取得了目前动作识别最好准确率。NTU RGB+D 和 UESTC 数据集都是室内视频数据集，为了检验作者的方法，他们又在野外的视频数据集 Kinetics 上做了one-shot测试，即每一类只选择一个样本进行训练，然后取得了很好的效果。

Introduction：首先作者介绍通常大家都使用卷积神经网络CNN来对视频数据集UFC101进行动作识别训练和预测，但是作者提出卷积神经网络非常依赖于数据集的大小，通常需要很大的数据集才能有好的效果，然后鉴于此就有很多工作
提出使用合成数据来增加数据量，例如使用光流估计，分割，身体和手势估计。在本文中研究的是利用合成的数据来进行动作识别。

作者通过观察，发现对于现在流行的所有网络，对于同一个动作，如果训练和测试都使用同一个视角，能得到很好的结果，但是如果训练和测试使用不同的视角，这些网络的性能就会大幅度减少。例如作者使用一个3D的卷积网络来对 NTU RGB+D 数据集进行训练，
当训练和测试都是正面视角的时候，最终能得到80%多的准确率，但是如果我们的测试换成90度视角，这个时候准确率就只有40%了。这个结果激发了我们来从一个巧妙的视角研究视频动作识别。

在之前有一些对人体姿势预测进行了研究，并且取得了很好的成绩，通常他们的目的是动作捕获（MoCap），所以这些研究不适合于行为的预测，因为它们没有数据标记。

所以本文就提出了一个新的简单有效的方法来合成带有行为标签的数据。首先我们使用 HMMR 和 VIBE 等方法来动态的从单视角的 RGB 图像中得到 3D 的人，这些 3D 的人是由一串 SMPL 的人体姿势的参数组成。
然后我们通过 SMPL 合成不同视角的带标签的训练数据。最后我们使用一个 3D 网络来对我们的数据进行训练，得到了非常好的效果。我们的效果主要有两个方面，一是对于没见过的视角的行为识别，二是对于 one-shot 数据的训练识别。

Related Work：      







 
[Learning from Synthetic Humans](https://arxiv.org/pdf/1701.01370.pdf) 