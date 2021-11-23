---
layout: blog
text-process: true
mathjax: true
background-image: http://7568.github.io/images/2021-11-03-transformer/img.png
category: 文本处理
title: 机器翻译 - Transformer
tags:
- Transformer
- 文本处理
---

[transformer-architecture]:http://7568.github.io/images/2021-11-03-transformer/transformer-architecture.png
[a-high-level-look]:http://7568.github.io/images/2021-11-03-transformer/a-high-level-look.png
[a-high-level-look-1]:http://7568.github.io/images/2021-11-03-transformer/a-high-level-look-1.png
[a-high-level-look-2]:http://7568.github.io/images/2021-11-03-transformer/a-high-level-look-2.png
[a-high-level-look-3]:http://7568.github.io/images/2021-11-03-transformer/a-high-level-look-3.png
[a-high-level-look-4]:http://7568.github.io/images/2021-11-03-transformer/a-high-level-look-4.png
[word-embedding]:http://7568.github.io/images/2021-11-03-transformer/word-embedding.png
[encoder-process]:http://7568.github.io/images/2021-11-03-transformer/encoder-process.png
[self-attention-process]:http://7568.github.io/images/2021-11-03-transformer/self-attention-process.png
[self-attention-process-2]:http://7568.github.io/images/2021-11-03-transformer/self-attention-process-2.png

# 简介
在文本处理中有两个经典的网络模型，一个是基于循环神经网络加上 attention 的 Seq2Seq 和完全基于 attention 的 Transformer。这两个模型在机器翻译中都取得了很好的效果。
本文中很大一部分内容来自翻译
[jalammar ： The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
，
[harvard ： The Illustrated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) 
和
[bentrevett ： Attention is All You Need](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb) 。

我们在上一篇[文章](https://7568.github.io/2021/11/03/seq2seqModel.html) 中讲述了 Seq2Seq with attention，也就是 [Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) 中的内容

Transformer 论文地址在 [Attention is All You Need.](https://arxiv.org/abs/1706.03762) 。

# 模型结构

首先 Transformer 还是经典的 encoder ，decoder 模型，不一样的地方在于 Transformer 没有使用 rnn 和 cnn 而是使用一种叫 self-attention 的技术，该技术相对于
rnn的优势是 self-attention 可以并行运算，从而使得大规模计算得以进行。不再是后面的单词需要等前面的单词运行完，得到前一个单词的 hidden 之后，再进行后面的运算。相对于 cnn 的优势是它是可解释的，
能够直观的看到翻译结果是由哪些因素决定的。

Transformer 整体结构如下：

![transformer-architecture]

我们从高层次来看该模型的化就是这样，一个输入，一个黑盒，一个输出

![a-high-level-look]

当我们拆开黑盒，就会发现里面包含两个模块，分别是 encoders 和 decoders

![a-high-level-look-1]

当我们继续探究黑盒，里面的 encoders 和 decoders ，我们就会发现，每一个 encoders 里面又包含有8个 encoder，decoders 里面也包含有8个 decoder。

![a-high-level-look-2]

当我们继续探究每一个 encoder ，就可以看到，每一个 encoder 都有相同的结构，都是由两部分构成，分别是 feed-forward neural network 和 self-attention 。

![a-high-level-look-3]

然后我们查看 decoder，可以看到，每一个 decoder 也是都包含相同的结构，都是由三部分构成，分别是 feed-forward neural network，Encoder-Decoder Attention，和self-attention 。

![a-high-level-look-4]

# 运行过程

现在我们大概了解了 transformer 的整体结构，接下来我们来看一看一个句子是如何从输入一步一步到输出的。

首先与常规的NPL处理一样，我们的输入都要经过 embedding 处理，将输入的每个单词变成向量。如下图所示

![word-embedding]

然后再放入到 encoder 里面，在一个 encoder 里面处理的流程如下：

![encoder-process]

接下来我们就来解释不同的单词是如何在 self-attention 中被处理，得到输出的。

## self-attention 介绍

首先假设我们有两个单词，分别是 Thinking，和 Machines。在计算 self-attention 之前首先要进行 embedding 运算，得到 <span style='color:#07d015;'> $$X_1 , X_2$$ </span>  ，
然后我们通过<span style='color:#07d015;'> $$X_1 , X_2$$ </span> 分别乘以矩阵<span style='color:#d436eb'>$$W^Q $$</span>,<span style='color:#ff8b00'>$$ W^K $$</span>,<span style='color:#30abff'>$$ W^V$$</span>，
得到<span style='color:#d436eb'>$$q_1 , q_2 $$</span>,<span style='color:#ff8b00'>$$ k_1 , k_2 $$</span>,<span style='color:#30abff'>$$ v_1 , v_2$$</span> ，他们分别表示为Querys，
keys，和Values。其中矩阵<span style='color:#d436eb'>$$W^Q$$</span> , <span style='color:#ff8b00'>$$W^K$$</span> , <span style='color:#30abff'>$$W^V$$</span>使用默认初始化数据，然后在训练过程中不断学习优化。整个过程如下图所示

![self-attention-process]

当我们得到了不同单词的<span style='color:#d436eb'>$$q$$</span> , <span style='color:#ff8b00'>$$k$$</span> , <span style='color:#30abff'>$$v$$</span>之后，我们就可以进行 self-attention 计算了。比如我们要计算<span style='color:green'>$$X_1$$</span>的self-attention结果，我们的操作流程如下：

![self-attention-process-2]

- 首先第一步就是计算得分，也就是图中的Score，Thinking 对自己的得分为<span style='color:#d436eb'>$$q_1$$</span> $$\times$$ <span style='color:#ff8b00'>$$k_1^T$$</span>，Thinking 对 Machines 的得分为<span style='color:#d436eb'>$$q_1$$</span> $$\times$$ <span style='color:#ff8b00'>$$k_2^T$$</span>，如果后面还有单词的化，计算得分为<span style='color:#d436eb'>$$q_1$$</span> $$\times$$ <span style='color:#ff8b00'>$$k_i^T$$</span>。
- 第二步将得分Score  除以 $$\sqrt{d_k}$$，$$d_k$$为$$k$$的维度，此处假设为8。
- 第三步为将第二步的结果进行 softmax 操作。
- 第四步将 softmax 的结果乘以各自的 Values，得到新的向量。
- 第五步将第四步的结果全部进行向量相加，得到一个新的向量<span style='color:#ff5ab2'>$$z_1$$</span>，这个<span style='color:#ff5ab2'>$$z_1$$</span>就是 Thinking 经过 self-attention 运算的结果。
- 当我们计算 Machines 的 self-attention 运算结果的时候，与 Thinking 流程是一样的，只是在计算 Score 的时候，使用的是<span style='color:#d436eb'>$$q_2$$</span>分别乘以<span style='color:#ff8b00'>$$k_1 , k_2 , ... , k_i$$</span>，来计算Thinking相对于各个单词的Score。剩下的流程其实是一样的。

更多参考来自于
- [graykode / nlp-tutorial](https://github.com/graykode/nlp-tutorial/blob/d05e31ec81d56d70c1db89b99ab07e948f7ebc11/5-1.Transformer/Transformer(Greedy_decoder).py)
- [Transformers: Attention in Disguise](https://www.mihaileric.com/posts/transformers-attention-in-disguise/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)