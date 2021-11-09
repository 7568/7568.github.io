---
layout: blog
text-process: true
background-image: http://7568.github.io/images/2021-11-03-seq2seqModel/2021-11-03_2.png
category: 文本处理
title: 机器翻译 - Seq2Seq with Attention
mathjax: true
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

## decoder

在 decoder 中，首先我们要对标签进行编码，然后，将编码后的结果放入到一个神经网络中，用来提取特征，

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

接下来我们对训练集的输入和输出进行 vocabulary 处理， vocabulary 处理其实就是找出输入输出中所有的单词，然后去重，然后给去重后的每个单词排序，得到每个单词的index，这个index就是每个单词的编码。

执行 vocabulary 处理代码如下

```python
# 构建词汇，构建之后，SRC 就多了个 vocab 属性，vocab 中包含有 freqs、itos、stoi 三个属性，其中freqs 表示的是 SRC 中每个单词和该单词的频数，也就是个数。
# itos 是一个列表，包含的的是频数 >= 2 的单词，stoi 用来标记 itos 中每个单词的索引，从0开始。
# 例如 输入是['two', 'two', ',', 'two', 'two', 'are', 'outside', 'near', 'many', 'bushes', 'two', 'young', ',', 'white', 'white', 'near', 'outside', 'near', 'many', 'bushes', '.']
# 则 freqs 是({'two': 5, 'near': 3, ',': 2, 'outside': 2, 'many': 2, 'bushes': 2, 'white': 2, 'are': 1, 'young': 1, '.': 1})
# itos 是 ['<unk>', '<pad>', '<sos>', '<eos>', 'two', 'near', ',', 'bushes', 'many', 'outside', 'white']
# 其中 <sos>：一个句子的开始，<eos>：一句话的结束，<UNK>: 低频词或未在词表中的词，比如我们设置 min_freq = 2 ，那么那些只出现了 1 次的单词，将来在放入到神经网络之前都会被 <UNK> 替换掉
# <PAD>: 补全字符，由于我们在进行批量计算的时候，每个样本的长度不一样，<PAD>就是用于保证样本长度一样的。参考于 https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15
# 由于我们的 min_freq = 2 ，所以可以看到频数为1的 'are'，'young'， '.' 都没有在 itos 中。
# stoi 是 {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3, 'two': 4, 'near': 5, ',': 6, 'bushes': 7, 'many': 8, 'outside': 9, 'white': 10})
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
# 输出一下结果
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
```

我们接下来看看我们的数据在进入到神经网络之前，都经历了怎样的处理 ：

- 首先是我们的原始数据，如下图展示。
  
  ![input-batch.png](http://7568.github.io/images/2021-11-03-seq2seqModel/input-batch.png)


- 接下来是我们的填充，首先在每句话的开始和结束分别加上'\<sos\>'和 '\<eos\>' ， 然后将整个 batch 中的数据对齐，按照最长的句子对齐，
  不够的用 '\<pad\>' 来填充，如下图所示。
  
  ![padded-input-batch.png](http://7568.github.io/images/2021-11-03-seq2seqModel/padded-input-batch.png)
  

- 最后就是将输入数据进行数字化处理，将每个单词分别转换成它所对应的索引，该索引就是 SRC.vocab stoi 中的值 ，如下图所示。
  
  ![input-numericalize.png](http://7568.github.io/images/2021-11-03-seq2seqModel/input-numericalize.png)

到此，我们的数据预处理就完成了。

接下来我们构造我们的 encoder 模型

在RNN系列中，传统的RNN存在比较大的梯度消失和梯度爆炸的问题，所以现在大家常常用LSTM来代替RNN，本文也将使用 LSTM 来进行编码
我们先看看LSTM的结构，该结构图来自于[dive into deep learning](https://d2l.ai/chapter_recurrent-modern/lstm.html)
![LSTM](http://7568.github.io/images/2021-11-03-seq2seqModel/lstm-struct.png)

最终的输出就是把 $$ H_t $$ 做一个线性变换，直接将 $$ H_t $$ 当作输出也是可以的。在 [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) 这篇文章中有关于 LSTM 更加详细的介绍。

于是我们的seq2seq模型就变成了如下结构，该图来自于 [nicolas-ivanov](https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15)

![seq2seq-lstm.png](http://7568.github.io/images/2021-11-03-seq2seqModel/seq2seq-lstm.png)

在原论文中作者使用的是4层LSTM，本文我们只使用2层LSTM进行训练。其中每层中又包含有多个LSTM单元，具体有多少个是根据输入的长度决定的。LSTM我们可以表示成如下表达式：

$$(h_t,c_t) = LSTM(e(x_t),h_(t-1),c_(t-1))$$

其中 $$h_t$$ 和 $$c_t$$ 分别表示第 t 个LSTM的输出中的隐藏单元和记忆单元，$$x_t$$ 表示第 t 个输入，$$e(x_t)$$ 表示将第 t 个输入进行 one-hot 处理。$$h_(t-1),c_(t-1))$$ 分别表示上一层的输出中的隐藏单元和记忆单元。
在理解上我们可以把 $$h_t$$ 和 $$c_t$$ 都当成隐藏单元，只不过计算方式不一样。** 其中 $$h_0$$ 和 $$c_0$$ ，是初始化随机生成的 ** 。

$$z^i = (h_l^i,c_l^i)$$

我们令 $$z^1$$ , $$z^2$$ 分别为每个隐藏单元的输出。$$z^i$$ 表示第 i 层的输出。$$h_l^i$$ 和 $$c_l^i$$ 表示第 i 层的最后一个LSTM单元的隐藏单元的输出和记忆单元的输出。

下图是一个LSTM编码的例子。其中黄色方块表示对输入进行 one-hot 处理，有2层绿色方块，表示有两层LSTM网络，每个绿色方块都表示一个LSTM单元，红色方块表示每层的输出。

![seq2seq2-encoder](http://7568.github.io/images/2021-11-03-seq2seqModel/seq2seq-encoder.png)

在 PyTorch 中，我们可以使用 nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout) 来创建一个LSTM网络，其中 ：

* emb_dim：输入的维度， 不是指一句话的长度，而是每个单词 one-hot 之后的向量的长度。
* hid_dim：隐藏单元的维度。
* n_layers：网络的层数，也是深度。
* dropout：每一层的 dropout。

**Note:** 需要注意的是，在LSTM中，如果我们的输入的维度只有1，那么我们就不能直接使用 nn.LSTM，而是使用 nn.LSTMCell，因为如果直接使用 nn.LSTM 会有维度转换的问题

代码如下：

````python

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell

````

我们再来构造我们的 decoder 模型

decoder 模型我们也是使用2层 LSTM ，原论文是4层。 网络结构跟 encoder 非常相似，是不过这里的 $$h_0$$ 和 $$c_0$$ 变成了 encoder 中的 $$z^1$$ , $$z^2$$ 。

下图展示的是我们的decoder的结构图

![seq2seq2-decoder](http://7568.github.io/images/2021-11-03-seq2seqModel/seq2seq-decoder.png)



接下来是 decoder 的代码实现：

````python

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

````



[code path](https://7568.github.io/codes/text-process/2021-11-03-seq2seqModel.py)



