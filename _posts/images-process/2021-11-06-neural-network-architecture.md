---
layout: blog 
images-process: true 
title: "神经网络结构"
background-image: https://7568.github.io/images/2021-11-06-neural-network-architecture/img.png
date:  2021-11-06 
category: 图像处理 
tags:
- 神经网络
- 网络结构
---

# 归一化方法

神经网络中结构中，归一化方法通常有BN(Batch Normalization)，LN(Layer normalization)，IN(Instance Normalization)，GN(Group Normalization)
，通常来说，归一化的目的是为了能让模型更快的收敛。

- BatchNorm：batch方向做归一化，跨样本、单通道，就是说一个batch中的不同的样本，相同的通道之间进行归一化。
- LayerNorm：channel方向做归一化，单样本、跨通道，就是说，在batch中，每一个样本，自己的通道一起做归一化。如果输入只有一个样本，计算到第i层的时候，输出为(C,H,W)，则LayerNorm就是对该输出整个做归一化。
- InstanceNorm：一个channel内做归一化，单通道，单样本，在batch中，每一个样本，不同的通道，单独自己做归一化。
- GroupNorm：先按channel方向分group，然后每个group内做归一化，然后将每一个group当作一个通道，进行LayerNorm操作。
  如果在分组的时候，组数为通道数，那么GroupNorm就与InstanceNorm等价。如果组数为1，那么GroupNorm就与LayerNorm等价。
  在网上找到这个人[amaarora](https://amaarora.github.io/2020/08/09/groupnorm.html) 的解释，挺详细的
  。最后还有个总结，说是在他的一个普通测试中，GN效果挺差，但是在[ Big Transfer (BiT): General Visual Representation Learning ](https://arxiv.org/abs/1912.11370)
  中，将 [ Weight Standardization ](https://arxiv.org/abs/1903.10520) 和GN结合使用过，会有很好的效果。
- GhostBatchNorm：该方法是将一个大的batch size 切分成多个小的 batch size，然后计算每个小的 batch 的 BatchNorm ，最后再把他们拼接起来。
  这个方法看起来很奇怪，但是有人将它和在网络中添加噪声进行和对比，发现结果差不多，
  所以该方法其实可以当作在网络中添加噪声来理解。该方法对于大 batch size 的时候有用。
  在 [Ghost BatchNorm explained](https://medium.com/deeplearningmadeeasy/ghost-batchnorm-explained-e0fa9d651e03) 这篇 blog
  中有关于GhostBatchNorm 的详细介绍和代码实现。

Bert LayerNorm 实现代码

```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # mean(-1) 表示 mean(len(x)), 这里的-1就是最后一个维度，也就是hidden_size维
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

# 防止梯度爆炸或者梯度消失

梯度爆炸和梯度消失在深度神经网络中是个很普遍的问题，只要网络很深，就一定会碰到该问题，以下是防止梯度爆炸和梯度消失的常用方法

- 重新设计网络，深层网络容易梯度爆炸和梯度消失，而浅层网络不容易梯度爆炸和梯度消失。
- 使用小的batch size。因为神经网络如果使用大的 batch size，在当他看到没见过的新数据的时候，往往会表现的并不好，所以此时会出现梯度爆炸。
- 使用权重归一化。
- 在循环神经网络中，将常规的RNN改成LSTM。一般可以防止梯度爆炸和梯度消失。
- 使用RELU或者类似RELU的激活函数。
- clip the gradients 能很好的防止梯度爆炸和梯度消失，其实就是让 gradients
  在更新的时候，锁定在一个区间内，防止梯度爆炸。通常在[Recurrent Neural Networks(RNN)](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)
  ，[Gated Recurrent Units (GRU)](https://d2l.ai/chapter_recurrent-modern/gru.html)
  ，[Long Short-Term Memory (LSTM)](https://d2l.ai/chapter_recurrent-modern/lstm.html) 等循环神经网络中用到。 clip the gradients
  的实现在pytorch中有`torch.nn.utils.clip_grad_value_()` , `torch.nn.utils.clip_grad.clip_grad_norm()`
  , `torch.nn.utils.clip_grad_norm_()`，他们的区别是 `torch.nn.utils.clip_grad_value_()` 将 grad
  限制在一个区间，`torch.nn.utils.clip_grad.clip_grad_norm()` 指先将grad进行归一化处理，然后再将grad限制在一个区间，`torch.nn.utils.clip_grad_norm_()`
  是 `torch.nn.utils.clip_grad.clip_grad_norm()` 的最新版，`torch.nn.utils.clip_grad.clip_grad_norm()` 是过期的版本。
- 使用残差网络能有效防止梯度爆炸和梯度消失，通常的做法是在网络中加入残差块
- 将卷积核参数进行正交规范化 orthogonal regularization