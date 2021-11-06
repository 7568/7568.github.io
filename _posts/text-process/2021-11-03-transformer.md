---
layout: blog
text_process: true
background-image: http://7568.github.io/images/2021-11-03_3.png
category: 文本处理，机器翻译
title: 机器翻译 - Transformer
tags:
- Transformer
- 文本处理
---

# 简介
在文本处理中两个经典的网络模型，一个是基于循环神经网络加上 attention 的 Seq2Seq 和完全基于 attention 的 Transformer。这两个模型在机器翻译中都取得了很好的效果。
本文中很大一部分内容来自翻译
[Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
和
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 。


本篇将主要讲述和翻译在 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
中的内容，Transformer 的相关论文地址在[Attention is All You Need.](https://arxiv.org/abs/1706.03762)
我们在上一篇[文章](https://7568.github.io/2021/11/03/seq2seqModel.html) 中讲述了 Seq2Seq with attention，也就是 [Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) 中的内容

代码资料主要来源于[哈佛大学NPL团队的一个blog](http://nlp.seas.harvard.edu/2018/04/03/attention.html)