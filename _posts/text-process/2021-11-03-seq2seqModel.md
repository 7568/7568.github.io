---
layout: blog
text-process: true
background-image: http://7568.github.io/images/2021-11-03-seq2seqModel/2021-11-03_2.png
category: 文本处理
title: 机器翻译 - Seq2Seq with Attention
tags:
- Seq2Seq
- Attention
- 文本处理
---

# 简介

在文本处理中两个经典的网络模型，一个是基于循环神经网络加上 attention 的 Seq2Seq 和完全基于 attention 的 Transformer。这两个模型在机器翻译中都取得了很好的效果。
本文中很大一部分内容来自翻译
[Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
和
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 。
代码参考于 [https://github.com/bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq)

本篇将主要讲述和翻译在[Seq2seq Models With Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
中的内容，Seq2seq的相关论文地址在[Sutskever et al.](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) , [Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)
我们将在下一篇[文章](https://7568.github.io/2021/11/03/transformer.html) 中讲述Transformer，也就是[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 中的内容

# 模型介绍

Sequence-to-sequence模型是一个深度学习神经网络模型，在很多像机器翻译，短文总结，和图像描述等任务中都取得了很好的成绩。接下来我将通过本blog来介绍 Seq2Seq 模型的相关内容和代码。希望对于初学者有所帮助。
本文主要讲述的代码是 Seq2Seq 模型在机器翻译上英文对中文的翻译。
<br/>
Seq2Seq 模型是典型的 encoder-decoder 模型，下面的动画将介绍 Seq2Seq 进行机器翻译时候的基本工作流程。左边是输入，右边是输出。

<video width="100%" height="auto" loop autoplay controls>
  <source src="http://7568.github.io/images/2021-11-03-seq2seqModel/2021-11-04_seq2seq_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

下面这个视频来自于 [https://github.com/google/seq2seq](https://github.com/google/seq2seq) 不过 `https://github.com/google/seq2seq`中的内容对本文关系不大

<video width="100%" height="auto" loop autoplay controls>
  <source src="http://7568.github.io/images/2021-11-03-seq2seqModel/2021-11-04_seq2seq_2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## encoder

在 Seq2Seq 模型的 encoder 中，要进行的工作有：

1. 将输入 X1 字符编码，变成数字类型，即 Word2Vec，得到 X1_Vec，如果我们的输入是 "早上好"，在 Word2Vec 中，先会加上开始标志 `<sos>` 和结束标志 `<eos>` ，
   这样输入就变成了5个字符，然后每个字符用一串0和1表示，于是得到5个Vector，就是我们想要的 X1_Vec 。
2. 将 X1_Vec 中的5个 Vector 依次按顺序放入到RNN中，得到一个输出 Z
   比如这样 ![encoder](http://7568.github.io/images/2021-11-03-seq2seqModel/2021-11-04_seq2seq_3.png)

### 代码实现

首先我们要安装pytorch(1.0以上)，torchtext，spacy

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

我们通过执行一下脚本来安装数据集

```shell
$ python -m spacy download en_core_web_sm
$ python -m spacy download de_core_news_sm
```

然后加载数据集

```python
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
```

接下来就是我们的编码阶段
首先我们将输入的一串连续的字符转换成list，如 '早上好' 转换成 '[早,上,好]' ，然后再将他们变成 0，1 编码，代码如下

```python
def tokenize_cn(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    将中文的一段话进行编码，将每个汉字都编码成一个字符串式的tokens。然后将他们反转，反转是为了放入RNN的时候保证最先放入的是一开始的字符，而不是最后的字符
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    英文同中文一样
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

在 torchtext 中已经有方法帮我们实现编码方法，我们只需要调用如下方法，分别进行中文和英文的编码

```python
SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
```

接下来我们加载数据集，然后自动分为训练数据，验证数据，和测试数据，本数据集使用的是 [Multi30k dataset](https://github.com/multi30k/dataset) ，
里面包含有 30000 条英文对法文和德文的句子。

```python
train_data, valid_data, test_data = Multi30k.splits(exts = ('.cn', '.en'),  fields = (SRC, TRG))
```

我们检查一下每个数据集的大小

```python
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")
```

查看一个样本数据，看看数据集的格式是什么样子的。

```python
print(vars(train_data.examples[0]))
```

我们得到的输出是这样子的，前面的 `src`中是德语，后面的 `trg`中是英语。

```json5
{'src': ['.', 'büsche', 'vieler', 'nähe', 'der', 'in', 'freien', 'im', 'sind', 'männer', 'weiße', 'junge', 'zwei'], 'trg': ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']}
```

接下来我们对训练集的输出和输出进行 vocabulary 处理， vocabulary 处理其实就是找出输入输出中所有的单词，然后去重，然后给去重后的每个单词排序，得到每个单词的index，这个index就是每个单词的编码
执行 vocabulary 处理代码如下

```python
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
# 输出一下结果
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
```

到此，我们的数据预处理就完成了。

接下来我们构造我们的 encoder 模型

在RNN系列中，传统的RNN存在比较大的梯度消失和梯度爆炸的问题，所以现在大家常常用LSTM来代替RNN，本文也将使用 LSTM 来进行编码
我们先看看LSTM的结构，该结构图来自于[dive into deep learning](https://d2l.ai/chapter_recurrent-modern/lstm.html)
![LSTM](http://7568.github.io/images/2021-11-03-seq2seqModel/2021-11-04_seq2seq_4.png)

最终的输出就是把 $$ h_t $$ [ h_t ]做一个线性. 换，直接将 $ h_t $ 当作输出也是可以的。
$$ h_t $$
最终的输出就是把 $$ h_t $$ [ h_t ]做一个线性. 换，直接将 $ h_t $ 当作输出也是可以的。

最终的输出就是把 $$ h_t $$ [ h_t ]做一个线性. 换，直接将 $ h_t $ 当作输出也是可以的。

$$ h_t $$

最终的输出就是把 $$ h_t $$ [ h_t ]做一个线性. 换，直接将 $ h_t $ 当作输出也是可以的。

