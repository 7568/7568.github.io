---
layout: blog
others: true
istop: true
title: "本blog搭建的方法"
background-image: http://7568.github.io/imaegs/2021-11-02_2.jpeg
date:  2021-11-02
category: others
tags:
- github
- blog
---

# 引言
最近做实验，一直没有好的结果，很是焦躁，又看到个医药的大公司招聘要求里面有说应聘者最好有个人技术专栏。为了缓解焦躁，就突发奇想，想着能写点东西，记点东西。
# 找模版
之前一直是有一个github的blog的，但是界面比较简单，这次又心血来潮，换了个新的，看上去还比较好看。最终找到了这个[liberxue][http://www.liberxue.com]
于是就拿过来改成自己的
# 改模版
1. 首先我们要找的模版一定要是 jekyll 的，这个是 github 自动支持的，也就是说当我们提交了脚本到 github 服务器之后，github 的系统会自动帮我们编译成网站的形式
[liberxue][http://www.liberxue.com] 就是 jekyll 的一个轻量级的blog模版
当我们下载了这个模版之后，首先我们需要到 `_config.yml` 文件中修改相应的内容，包括本bblog的作者信息，其中最主要是修改 `links` 中的内容，当然，如果你的内容跟原作者的一样，就不需要修改
`links` 中的内容就是网页的头的部分的修改，比如我修改之后是这个样子的![head](../../imaegs/2021-11-02_1.png)

2. 接下来就是修改原来目录中的文件夹的名字，修改成`_config.yml`中`links`部分一一对应。比如我将`_config.yml`中`links`中的 book 改成了 images_process ，
   那么我就要把原来的 book 目录的名字改成 images_process ，然后在 images_process 中的 `index.html`文件中第10行的``` {% if posts.book %} ```
   改成``` {% if posts.images_process %} ```。其余的都按照这个修改。
   
3. 在 _posts 文件夹中创建多个文件夹，用于存放不同类型的blog，_posts 中的文件夹的名字可以随便取。

4. 最后，当我们要写blog的时候，我们需要在我们的blog中写入一下内容，其中最重要的内容过就是layout的下面一行，它决定着本blog的分类，
   比如我这个blog的分类是'其他'，那么就把`others`设置为ture，这样当我们提交了本blog之后，github把我们的网站编译后就自动分到了对应的类别中。

```yaml
---
layout: blog
others: true
istop: true
title: "本blog搭建的方法"
background-image: http://7568.github.io/imaegs/2021-11-02_2.jpeg
date:  2021-11-02
category: others
tags:
- github
- blog
---
```

5. 原来的blog中有很多原作者的信息，文字内容，全部修改成自己的就行了，换一下'style/favicons/favicon.ico'和
   'style/images/logo-liberxue.png.ico'图片，搜索整个目录，将'liberxue'改成自己的名字就可以了。

