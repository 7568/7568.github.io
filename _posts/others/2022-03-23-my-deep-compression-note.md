---
layout: blog
others: true
istop: true
title: "deep compression note"
background-image: http://7568.github.io/images/2022-03-23-my-deep-compression-note/img.png
date:  2022-03-23
category: 杂记
tags:
- github
- blog
---

[figure_1]:https://7568.github.io/images/2022-03-23-my-deep-compression-note/figure_1.png

(structured pruning) Anwar et al. (2015) describe structured pruning in convolutional layers at the
level of feature maps and kernels, as well as strided sparsity to prune with regularity within kernels.

(unstructured pruning) Han et al. (2015) introduce a simpler approach by fine-tuning with a strong $$\ell_2$$ regularization term
and dropping parameters with values below a predefined threshold. Such unstructured pruning is very
effective for network compression, and this approach demonstrates good performance for intra-kernel
pruning. But compression may not translate directly to faster inference since modern hardware exploits regularities in computation for high throughput. So specialized hardware may be needed
for efficient inference of a network with intra-kernel sparsity (Han et al., 2016).This approach
also requires long fine-tuning times that may exceed the original network training by a factor of
3 or larger.

Group sparsity based regularization of network parameters was proposed to penalize
unimportant parameters (Wen et al., 2016; Zhou et al., 2016; Alvarez & Salzmann, 2016; Lebedev
& Lempitsky, 2016). Regularization-based pruning techniques require per layer sensitivity analysis
which adds extra computations

combining parameters with correlated weights (Srinivas & Babu, 2015),
reducing precision (Gupta et al., 2015; Rastegari et al., 2016) or tensor decomposition (Kim et al.,
2015). 

（https://arxiv.org/pdf/1611.06440.pdf） propose a new scheme for iteratively pruning deep convolutional neural networks ，and  in its iterative
procedure they only remove unimportant parameters leaving others untouched

[A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/pdf/1710.09282.pdf) 中指出深度压缩总共可以分成4类，分别是
（1）参数剪枝和量化 （parameter pruning and quantization） ， （2）低秩分解（low-rank factorization），（3）转换或者压缩卷积核（transferred/compact convolutional filters） ， （4）知识蒸馏（knowledge distillation. ）

参数剪枝：设计一个策略（如设置一个阈值，或者通过一个方法来计算参数的重要性），让网络中的连接数变少，即将原来训练好的参数，一部分变成0，这样网络参数就变得稀疏，从而使用稀疏矩阵的方法来保存参数。

参数量化：（1）将原来高精度的网络参数用低精度的字节长度来表示，例如原来是32位bit，现在换成16位，就能节约参数所需的存储空间，甚至有人直接使用2位的bit来表示参数。
（2）在训练的时候将参数矩阵设计成方便存储的特殊结构，从而实现减少参数所需的存储空间。例如设置一个斜对角线矩阵结构作为参数矩阵的结构。

低秩分解：使用矩阵分解的思想将已经训练好了的参数进行分解，如svd分解。这样就可以使用一个参数较少的矩阵来代替原来的矩阵。

转换或者压缩卷积核：这种方法是一种常见的方法，例如利用卷积操作代替全连接，或者使用较小的卷积核来代替较大的卷积核，或者maxpooling，或者使用别的卷积方法。

知识蒸馏：首先我们使用常规方法获得一个模型，称为老师模型，这个模型的训练数据是标记好了的数据，然后我们再设计一个小的模型，称为学生模型，使用未标记的数据来进行训练。对于一个未标记的数据，分别经过老师模型和学生模型，使用老师模型来对数据进行标记，从而得到损失，进行训练。

*idea* : 在[Do Deep Nets Really Need to be Deep?](https://papers.nips.cc/paper/2014/file/ea8fcd92d59581717e06eb187f10666d-Paper.pdf)这篇论文中说使用一个训练好了的老师模型来训练学生模型，得到的结果最终学生模型比老师模型效果要好。但是这篇
论文中的学生模型使用的是更宽的模型。我们可以使用[Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf)这篇论文里面的方法来获得学生模型，即将剪枝之后的模型当作学生模型。在原论文中剪枝之后的模型通常
精度与原始模型一样，如果我们在retrain剪枝模型的时候，使用一个老师模型来训练，就可能获得一个更好的效果。

[Recent Advances in Efficient Computation of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1802.00939.pdf) 这篇论文也是一篇关于深度网络压缩的综述性论文。里面分别介绍了6种网络压缩和加速的类型，分别是：网络剪枝（network pruning） ， 低秩近似（ low-rank
approximation），网络量化（network quantization），老师学生网络（teacher-student networks），紧凑型网络设计（compact network design）和硬件加速器（Hardware Accelerator）。
其中，在网络剪枝方面，根据剪枝的粒度不同，细分了5类来分别描述，分别是： 细粒度剪枝（ fine-grained pruning） ， 向量级别的剪枝（vector-level pruning），核级别的剪枝（kernel-level pruning），祖级别的剪枝（group-level pruning ），过滤器级别的剪枝（ filter-level pruning）。<br/>
细粒度剪枝（ fine-grained pruning） 指的是以一种非结构化的方式来剪枝，选定一个判别方式，然后通过这个判别方式来判断某个参数是否要剪掉。<br/>
向量级别的剪枝（vector-level pruning）这个方法相对于 fine-grained pruning 来说，更加结构化一些，他是通过设计一个判别方法来判断卷积核中的某一个向量的参数是否需要被剪枝掉。<br/>
核级别的剪枝（kernel-level pruning）这个方法跟 vector-level pruning 类似，只是这个方法剪枝的是矩阵。<br/>
组级别的剪枝（group-level pruning ）这个方法指的是在不同的卷积核上，设计一个判别方式，然后大家剪同样部分的枝。<br/>
过滤器级别的剪枝（ filter-level pruning）这个方法指的是直接剪切掉某一个卷积核，这个方法的颗粒度最大 <br/>
下图是不同的颗粒度剪枝方法的图示：
![figure_1]

低秩近似（ low-rank approximation ）<br/>
这个方法分成三类，分别是 two-component decomposition, three-component decomposition and four-component decomposition。two-component decomposition通常是SVD分解，然后three-component decomposition和four-component decomposition是将SVD分解之后的矩阵再进行分解成更小数量更多的矩阵。

网络量化（ Network Quantization ）<br/>
网络量化分成两种，一种是标量和矢量的量化，另一种权重的固定点量化。标量的方法指的是使用一个值来代替一组值，从而使得网络参数变小，例如使用 Kmeans 来进行分组，然后用中心值来代替一组值。
矢量的方法指的是我们最终使用一个或者若干个向量来代替一个矩阵，从而减少矩阵存储所用的空间。权重的固定点量化指的是通常我们的权重都是使用的32bit的数据长度来保存的，但是有研究认为其实是不需要这么大的数据精度的，于是就有人提出了使用16bit，8bit，3bit，2bit的方式来训练和保存权重参数。
权重的固定点量化又可以分成两类，一类是直接对权重进行量化，一类是对激活函数的运算进行量化。也可以两个一起使用。

老师学生网络（Teacher-student Network）<br/>
老师学生网络说的是我们可以使用一个大的已经训练好了的老师网络来训练一个小的学生网络，而且通常学生网络表现好于老师网络，具体的实现细节就是对于一个味标记的数据，先用老师网络得到预测结果，然后用学生网络来进行预测，并通过该结果与老师网络的结果计算损失，从而对学生网络进行优化。

紧凑型网络设计（Compact Network Design）<br/>
这个方法指的是我们使用小的卷积核来代替大的卷积核，从而来减少网络的参数量。例如使用1x1的卷积核。

硬件加速器（Hardware Accelerator）<br/>
硬件加速器分成两类，一类是加快运算速度，一类是减少运算所需的能源消耗。

[COMPRESSING DEEP CONVOLUTIONAL NETWORKS USING VECTOR QUANTIZATION](https://arxiv.org/pdf/1412.6115.pdf) 这篇文章主要介绍了4种神经网络参数量化的方法，分别是：BINARIZATION，SCALAR QUANTIZATION USING kMEANS， PRODUCT QUANTIZATION， RESIDUAL QUANTIZATION。<br/>
其中 BINARIZATION 指的是将矩阵中的数据全部变成1和-1，大于0的变成1，小于0的变成-1。<br/>
SCALAR QUANTIZATION USING kMEANS 指的是使用 kmeans 方法，将矩阵分成k类，然后用中心值来代替一组值。<br/>
PRODUCT QUANTIZATION 指的是对于一个矩阵，首先我们选择某一种方式，按照列分成s个小的矩阵，然后分别对每个小的矩阵按照行进行Kmeans聚类，分成k类，这样我们就可以使用一个向量来表示一个小矩阵，从而就使用s个向量来表示一个矩阵。<br/>
RESIDUAL QUANTIZATION 指的是首先使用kmeans将矩阵进行分类，分成k个向量组，得到k个中心点的向量，然后我们计算每个剩余向量与中心向量的残差，从新得到k个分组。也就是说我们使用kmean来确定k个中心，然后利用计算残差来得到每一组的元素。

