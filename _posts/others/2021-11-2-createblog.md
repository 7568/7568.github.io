---
layout: blog
others: true
istop: true
title: "本blog搭建的方法"
background-image: http://7568.github.io/images/2021-11-2-createblog/2021-11-02_2.jpeg
date:  2021-11-02
category: 杂记
tags:
- github
- blog
---

### 引言
最近做实验，一直没有好的结果，很是焦躁，又看到个医药的大公司招聘要求里面有说应聘者最好有个人技术专栏。为了缓解焦躁，就突发奇想，想着能写点东西，记点东西。

### 找模版
之前一直是有一个github的blog的，但是界面比较简单，这次又心血来潮，换了个新的，看上去还比较好看。最终找到了这个[Liberxue](http://www.liberxue.com)，
于是就拿过来改成自己的

### 改模版
1. 首先我们要找的模版一定要是 jekyll 的，这个是 github 自动支持的，也就是说当我们提交了脚本到 github 服务器之后，github 的系统会自动帮我们编译成网站的形式，
[Liberxue](http://www.liberxue.com) 就是 jekyll 的一个轻量级的blog模版，
当我们下载了这个模版之后，首先我们需要到 `_config.yml` 文件中修改相应的内容，包括本bblog的作者信息，其中最主要是修改 `links` 中的内容，当然，如果你的内容跟原作者的一样，就不需要修改，
`links` 中的内容就是网页的头的部分的修改，比如我修改之后是这个样子的![head](http://7568.github.io/images/2021-11-2-createblog/2021-11-02_1.png)

2. 接下来就是修改原来目录中的文件夹的名字，修改成`_config.yml`中`links`部分一一对应。比如我将`_config.yml`中`links`中的 book 改成了 images_process ，
   那么我就要把原来的 book 目录的名字改成 images_process ，然后在 images_process 中的 `index.html`文件中第10行的
   `{% if posts.book %} ... {% endif %}` 改成 `{% if posts.images_process %} ... {% endif %}`  其余的都按照这个修改。
   
3. 在 _posts 文件夹中创建多个文件夹，用于存放不同类型的blog，_posts 中的文件夹的名字可以随便取。

4. 最后，当我们要写blog的时候，我们需要在我们的blog开头部分写入以下内容，其中最重要的内容过就是layout的下面一行，它决定着本blog的分类，
   比如我这个blog的分类是'其他'，那么就把`others`设置为ture，这样当我们提交了本blog之后，github把我们的网站编译后就自动分到了对应的类别中。

   ```
   ---
   layout: blog   表示当前文章的排版方式，通常都是blog
   others: true   表示当前文章的所属分类，比如如果是其他杂记分类，那么就设置为others:true，如果是图像处理就设置为images_process:true
   istop: true    是否置顶，如果是的话，就会在首页的最下面一直显示，即使时间过去了很久，由于我把该网站的页脚删了，所以这个没用了
   title: "本blog搭建的方法" 本文的标题
   background-image: http://7568.github.io/imaegs/2021-11-02_2.jpeg   本文在展示页面中显示的图像
   date:  2021-11-02 本文的编辑时间
   category: others 本文的分类，这个可以是任意分类，只会在展示页面中显示出'others'来，只是展示作用，不是真正的分类，最好与网页的头上的中文名一样，方便显示，且不要太长，否则
   mathjax: true md页面中是否需要显示数学表达式，如果有就为true
   tags:  本文的标签
   - github
   - blog
   ---
   ```
   页面显示与设置的关系如下图所示![image 1](https://7568.github.io/images/2021-11-2-createblog/2021-11-02-createblog_1.png)

5. 原来的blog中有很多原作者的信息，文字内容，全部修改成自己的就行了，换一下'style/favicons/favicon.ico'和
   'style/images/logo-liberxue.png.ico'图片，搜索整个目录，将'liberxue'改成自己的名字就可以了。
   
*Note:* 切记文章的 title 和 category 中一定不要有 ':' ，不要有冒号！！！

