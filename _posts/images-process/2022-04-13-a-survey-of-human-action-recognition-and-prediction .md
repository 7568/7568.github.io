---
layout: blog
others: true
istop: true
title: "action recognition and prediction"
background-image: http://7568.github.io/images/2022-04-13-a-survey-of-human-action-recognition-and-prediction/img.png
date:  2022-04-13
category: 视频动作识别综述
tags:
- survey
- video
- action recognition
- action prediction
---

[figure_1]:https://7568.github.io/images/2022-04-09-synthetic-humans-for-action-recognition/figure_2.png
[figure_2]:https://7568.github.io/images/2022-04-09-synthetic-humans-for-action-recognition/figure_1.png



[Human Action Recognition and Prediction: A Survey](https://arxiv.org/pdf/1806.11230.pdf)

# 摘要
在本文中，我们研究了目前最好的视频动作识别和预测的模型。包括模型，算法，技术困难，流行的行为数据库，评价准则，有前途的未来方向，都进行了系统的讨论。

# Introduction
人工智能的一个终极目的是建造一个机器，它能精准的理解人类的行为和意图，这样它就能很好的服务于我们。常见的人类行为任务有：（1）个人的行为，例如鼓掌，奔跑。
（2）人类交互，例如握手。（3）人物交互，例如体育运动。（4）集体行为。（5）带有景深的行为。（6）多视角行为。

基于视觉的人类行为任务主要分为两类，分别是：（1）行为识别，观看一段完整的行为视频后，判断视频中的动作。（2）行为预测，观看一段完整动作一部分的视频，判断动作。

行为识别是一个基本的任务，通常该任务需要考虑行为的表示和行为的分类。行为预测智能看到一部分的数据，就需要判断接下来的动作。行为识别和预测的一个最主要
的区别是做出判断的时间。

## Real-World Applications

### Visual Surveillance
### Video Retrieval
在视觉监控中，一个重要的任务是视频内容的抽取，比如我们需要从一段很长的视频中抽取出关键的一部分，然后做行为判断。
### Entertainment
###Human-Robot Interaction
### Autonomous Driving Vehicle

## Research Challenges

#### Intra- and inter-class Variations
内部不确定性包括同一动作从不同的视角观察，同一动作由不同的人来做，还有就是一些不同的动作的相似性，例如走和跑就很相似。