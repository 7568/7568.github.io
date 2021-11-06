---
layout: blog
images-process: true
title: "神经网络中的各种卷积操作"
background-image: https://7568.github.io/images/2021-11-06-conv_arithmetic/no_padding_no_strides.gif
date:  2021-11-06
category: 图像处理
tags:
- 神经网络
- 卷积
---
本文大部分内容来自于 [vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) 

# Convolution arithmetic

A technical report on convolution arithmetic in the context of deep learning.

The code and the images of this tutorial are free to use as regulated by the 
licence and subject to proper attribution:

* \[1\] Vincent Dumoulin, Francesco Visin - [A guide to convolution arithmetic
  for deep learning](https://arxiv.org/abs/1603.07285)
  ([BibTeX](https://gist.github.com/fvisin/165ca9935392fa9600a6c94664a01214))

## Convolution animations

_N.B.: Blue maps are inputs, and cyan maps are outputs._

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/no_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/arbitrary_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/same_padding_no_strides.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
    <td>Full padding, no strides</td>
  </tr>
  <tr>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/no_padding_strides.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/padding_strides.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/padding_strides_odd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
    <td></td>
  </tr>
</table>

## Transposed convolution animations

_N.B.: Blue maps are inputs, and cyan maps are outputs. (转置卷积)_

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/no_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/same_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/full_padding_no_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Half padding, no strides, transposed</td>
    <td>Full padding, no strides, transposed</td>
  </tr>
  <tr>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/no_padding_strides_transposed.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/padding_strides_transposed.gif"></td>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/padding_strides_odd_transposed.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides, transposed</td>
    <td>Padding, strides, transposed</td>
    <td>Padding, strides, transposed (odd)</td>
    <td></td>
  </tr>
</table>

按照[dive into deep learning](https://d2l.ai/chapter_computer-vision/transposed-conv.html) 中的描述，转置卷积相对于普通卷积的区别就是，普通卷积是在一次卷积中，卷积核乘以相应大小的输入之后，会相加，最后变成一个输出，而转置卷积则是卷积核与输入中的一个值相乘，得到一个与卷积核大小一致的输出。
![img_2.png](https://7568.github.io/images/2021-11-06-conv_arithmetic/transposed-convolution.png)
一次反卷积计算中，0先跟kernel相乘，得到[[0,0],[0,0]]，变成 2x2 的矩阵，并没有再相加为一个数，如如果是普通卷积操作，就会再相加成一个数

下面是带步长的转置卷积![img_1.png](https://7568.github.io/images/2021-11-06-conv_arithmetic/transposed_convolution_with_stride_2.png)
转置卷积通常能够使得输出变大，在U型网络中常常被用到
转置卷积通常也叫做反卷积，因为，普通的卷积操作，为了将卷积操作转换成矩阵相乘，在计算机中处理的时候是这个样子的 ( 图片来自于[机器人博士 【图解】卷积和反卷积过程Convolution&Deconvolution](https://zhuanlan.zhihu.com/p/52407509) )  
![计算卷积](https://7568.github.io/images/2021-11-06-conv_arithmetic/convolution_in_conputation.png)

而在计算机计算转置卷积的时候，在计算机中处理的时候是这个样子的  ( 图片来自于[机器人博士 【图解】卷积和反卷积过程Convolution&Deconvolution](https://zhuanlan.zhihu.com/p/52407509) )  
![计算反卷积](https://7568.github.io/images/2021-11-06-conv_arithmetic/transposed_convolution_conputation.png)
所以称转置卷积也叫反卷积
## Dilated convolution animations

_N.B.: Blue maps are inputs, and cyan maps are outputs. (Dilated: 膨胀的；扩张的)_

<table style="width:25%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="https://7568.github.io/images/2021-11-06-conv_arithmetic/dilation.gif"></td>
  </tr>
  <tr>
    <td>No padding, no stride, dilation</td>
  </tr>
</table>

## Generating the Makefile

From the repository's root directory:

``` bash
$ ./bin/generate_makefile
```
## Generating the animations

From the repository's root directory:

``` bash
$ make all_animations
```

The animations will be output to the `gif` directory. Individual animation steps
will be output in PDF format to the `pdf` directory and in PNG format to the
`png` directory.

## Compiling the document

From the repository's root directory:

``` bash
$ make
```

以上内容全部过来自于[vdumoulin / conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)