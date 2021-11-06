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
- BatchNorm：batch方向做归一化，跨样本、单通道，就是说一个batch中的不同的样本，相同的通道之间进行归一化
- LayerNorm：channel方向做归一化，单样本、跨通道，就是说，在batch中，每一个样本，自己的通道一起做归一化。如果输入只有一个样本，计算到第i层的时候，输出为(C,H,W)，则LayerNorm就是对该输出整个做归一化。
- InstanceNorm：一个channel内做归一化，单通道，单样本，在batch中，每一个样本，不同的通道，单独自己做归一化
- GroupNorm：先按channel方向分group，然后每个group内做归一化，然后将每一个group当作一个通道，进行LayerNorm操作。
  如果在分组的时候，组数为通道数，那么GroupNorm就与InstanceNorm等价。如果组数为1，那么GroupNorm就与LayerNorm等价
  在网上找到这个人的解释[amaarora](https://amaarora.github.io/2020/08/09/groupnorm.html) ，挺详细的 。最后还有个总结，说是在他的一个普通测试中，GN效果挺差，但是在[ Big Transfer (BiT): General Visual Representation Learning ](https://arxiv.org/abs/1912.11370) 
  中，将 [ Weight Standardization ](https://arxiv.org/abs/1903.10520) 和GN结合使用过，会有很好的效果
Bert layer norm实现代码
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
