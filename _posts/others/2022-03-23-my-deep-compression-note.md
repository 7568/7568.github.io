---
layout: blog
others: true
istop: true
title: "deep compression note"
background-image: http://7568.github.io/images/2021-11-2-createblog/2021-11-02_2.jpeg
date:  2022-03-23
category: 杂记
tags:
- github
- blog
---

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



