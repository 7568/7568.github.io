---
layout: post
title: Install and use jenkins .
---

Yestaday, I have finished the small java web project , so i want to deploy in server , so i want to use jenkins to deploy it , before now, i just to use jenkins ,  and never install it by myself, so this time I want to install jenkins .

1. install jenkins war parckeg from [jenkins](https://jenkins.io)
    ![images help](/images/008.png)
2. move this war parckage to you servers
3. run this command `java -jar jenkins.war --httpPort=9876`  , then you will look this 
    ![images help](/images/010.png)
4. browse to `http://serverip:9876`
5. then you will see this
    ![images help](/images/009.png)
6. input the init  passport ,and choose suggest plugins ,then wait it automation install plugin
7. after plugins have installed ,you will see this page 
    ![images help](/images/011.png)
8. create first admin user , then Jenkins is ready! then click the button *Start using Jenkins*
    ![images help](/images/012.png)
9. now you can use it.
10. now create a new project ,input project name , and my code is in svn , so I use Subversion, like this:
  ![images help](/images/013.png)
11. then I use maven to build my project , so I use this `mvn clean package spring-boot:repackage -f pom-pro.xml` command shell
12. after build project , now I want to deploy it to tomcat,to i need a plugin [    
Deploy to container Plugin](https://wiki.jenkins-ci.org/display/JENKINS/Deploy+Plugin)
13. after install plugin now go back to you project settings page , add this
   ![images help](/images/014.png)
   ![images help](/images/015.png)
14. then input this and goto you tomcat server,check if the manager project is exist, if not goto download and remove it to tomcat's webapps directory,and change the tomcat conf directory , vi the tomcat-users.xml,
add this code.
    ```
    <role rolename="manager-script"/>
    <user username="deployer" password="deployer" roles="manager-script" />
    ```
15. then restart you tomcat ,and go back to jenkins ,click the build button,then go to Console Output,check the build status,if no accident , it will be success. 