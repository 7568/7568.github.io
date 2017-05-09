---
layout: post
title: nginx 指定文件夹的访问
---

想通过nginx做一个http的文件夹访问服务器,在网上找了一些资料，自己在这里记录一下，方便以后使用

- 在config文件中加入

```
location /tomcat/ {
            alias   D:/log/;
        }
```
- *注意*路劲中不能带有空格
- 如果想让带有空格的文件夹也能访问，那就弄成下面这样，其实就是加上引号，哈哈，我之前为这个弄了好久
```
location /tomcat/ {
            alias   "D:\Program Files (x86)\\apache-tomcat-7.0.77\webapps\\";
        }
```