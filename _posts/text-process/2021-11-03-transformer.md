---
layout: blog
text-process: true
background-image: http://7568.github.io/images/2021-11-03-transformer/2021-11-03_3.png
category: 文本处理
title: 机器翻译 - Transformer
tags:
- Transformer
- 文本处理
---

# 简介
在文本处理中有两个经典的网络模型，一个是基于循环神经网络加上 attention 的 Seq2Seq 和完全基于 attention 的 Transformer。这两个模型在机器翻译中都取得了很好的效果。
本文中很大一部分内容来自翻译
[jalammar ： The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
，
[harvard ： The Illustrated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) 
和
[bentrevett ： Attention is All You Need](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb) 。

我们在上一篇[文章](https://7568.github.io/2021/11/03/seq2seqModel.html) 中讲述了 Seq2Seq with attention，也就是 [Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) 中的内容

[Attention is All You Need.](https://arxiv.org/abs/1706.03762)

更多参考来自于
- [graykode / nlp-tutorial](https://github.com/graykode/nlp-tutorial/blob/d05e31ec81d56d70c1db89b99ab07e948f7ebc11/5-1.Transformer/Transformer(Greedy_decoder).py)