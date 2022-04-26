---
layout: blog
images-process: true
istop: true
title: "action recognition and prediction"
background-image: http://7568.github.io/images/2022-04-13-a-survey-of-human-action-recognition-and-prediction/img.png
date:  2022-04-13
category: 动作识别综述
tags:
- survey
- video
- action recognition
- action prediction
---

[figure_1]:https://7568.github.io/images/2022-04-09-synthetic-humans-for-action-recognition/figure_2.png
[figure_2]:https://7568.github.io/images/2022-04-09-synthetic-humans-for-action-recognition/figure_1.png
[31. ](#abcd)


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
内部不确定性包括同一动作从不同的视角观察，同一动作由不同的人来做，还有就是一些不同的动作的相似性，例如走和跑就很相似 。这些都会混淆很多行为识别的算法。

### Cluttered Background and Camera Motion
很多算法对于室内的视频动作识别有效，但是对于复杂环境的室外的动作视频却效果很差。主要是因为室外视频中背景的杂质。相机的移动也是阻碍现实世界的动作识别的一个因素。
对于一些相机移动幅度很大的视频，动作的特征提取会变得相当困难。所以就有人研究用专门的模型和惩罚措施来解决相机移动问题。其他的与环境相关的问题还有照明的情况，视角的改变，动态背景，等
也是视频动作识别的挑战。

### Insufficient Annotated Data
虽然现在有一些视频动作识别的方法在小规模数据集上表现不错，但是这些方法都是针对室内情况的，对于室外的复杂环境，还是很大的挑战。最近深度学习的方法对于那些
非受控的环境中的数据集上表现出很好的效果。但是这些方法都需要很大的标记数据。像 MDB51 和 UCF-101 包含有几千个视频，但是对于训练一个深度网络还远远不够。
虽然Youtube-8M 和 Sposrts-1M 这两个数据集包含有百万个动作视频，但是他们的标记可能是不准确的。在这样的数据集上训练一个神经网络可能是不可靠的。然而有一种可能的方法是
有一些标记的数据是可用的，这样就可以在训练的时候将标记数据和未标记数据混合起来使用，所以现在一个很必要的任务是设计一个能同时从标记数据和未标记数据中学习的算法。

### ction Vocabulary
人类行为能够分成不同的级别，动作，行为最小单元，组合行为，事件等等，这就定义了一个行为的层级结构，在高级的层级中复杂的行为可以通过低层级的简单行为合成而来。于是如何定义和分析不同种类的行为就变得极为重要。

### Uneven Predictability
在视频动作识别中，不同的帧对于最终的影响的不一样的。有一些研究显示，一个视频能够通过一些帧有效的表示。这表明在一个视频中大部分的帧是多余的。
于是就有一些研究中的方法要求输入的数据一开始的帧就是关键帧。而且有一些动作只需要很少的帧就很容易的识别出来，而有一些动作需要很多的帧才能正确识别。但是现实的需求是对于任何的动作，
我们最好是能很快的通过少数的帧就能正确识别。

# Human Perception of Actions
人类的行为，特别是那些涉及到整个肢体和身体的动作，还有那些与环境的交互，都包含了表演者的意图，目标，精神状态等丰富的信息。理解别人的行为和意图是一项很重要的
社交技能，而人来的视觉系统为这个技能提供了丰富的资源信息。相较于静态的图像，视频中的人类行为能够提供更可靠和更深刻的信息。而且里面的人说的话也能比图像更能让人理解别人的动作。


























<a name="abcd"> Clarke, T., Bradshaw, M., Field, D., Hampson, S., Rose, D.: The perception of emotion from body 
movement in point-light displays of interpersonal dialogue. Percep- tion 24, 1171–80 (2005)</a>
