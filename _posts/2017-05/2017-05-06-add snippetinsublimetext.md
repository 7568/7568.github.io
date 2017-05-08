---
layout: post
title: add snippet in sublimetext like as input image quickly or insert default layout .
---

sometimes I want to insert somecode automation like as insert image model when I code the markdown,so I can write a little code to compelete this feature , in sublimetext I just add snippet , it's easy

fist just open a new snippet like this
 ![images help](/images/005.png)
ad then you will look like this :
 ![images help](/images/006.png)
change the codo like this:
 ![images help](/images/007.png)
the code is 
```
<snippet>
	<content><![CDATA[
![${1:images help}](/images/${2:image}.png)
]]></content>
</snippet>
```
 ok , now add shortcut to show snippet
 just add this code 
```
{"keys": ["ctrl+y"], "command": "show_overlay", "args": {"overlay": "command_palette", "text": "Snippet: "} }
```
to you Key Bingdings ,now everything be done
---
The insert image snippet code is 
```
<snippet>
    <content><![CDATA[
![${1:images help}](/images/${2:image}.png)
]]></content>
</snippet>
```