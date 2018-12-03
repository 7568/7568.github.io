安装nginx-1.8.0

1. 下载nginx的包，地址：http://nginx.org/download/nginx-1.8.0.tar.gz

2. 下载依赖的pcre，地址：ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre-8.41.tar.gz

   1. 安装

      ```shell
      sudo tar -zxvf pcre-8.41.tar.gz
      cd pcre-8.41
      sudo ./configure
      sudo make
      sudo make install
      ```

      ​

3. 下载依赖的OpenSSL ，yum install openssl

4. 下载依赖的zlib，地址：https://www.zlib.net/fossils/zlib-1.2.11.tar.gz

   1. 安装

      ```shell
      sudo tar -zxvf zlib-1.2.8.tar.gz
      cd zlib-1.2.8
      sudo ./configure
      sudo make
      sudo make install
      ```

      ​

5. 安装nginx

   ```shell
   ./configure --sbin-path=/home/root/nginx/nginx  --conf-path=/home/root/nginx/nginx.conf --pid-path=/home/root/nginx/nginx.pid --with-http_ssl_module --with-pcre=../pcre-8.41 --with-zlib=../zlib-1.2.11
   ```

   ​

6. 测试

   1. 启动

      ```shell
      ./nginx
      ```

   2. 停止

      ```shell
      pkill nginx
      ```

      ​

7. 设置开机自动启动

   ```shell
   cd /etc/init.d
   touch nginx
   chmod a+x nginx
   vi nginx
   ```

   ```properties
   #!/bin/bash
   # nginx Startup script for the Nginx HTTP Server
   # it is v.0.0.2 version.
   # chkconfig: - 85 15
   # description: Nginx is a high-performance web and proxy server.
   #              It has a lot of features, but it's not for everyone.
   # processname: nginx
   # pidfile: /var/run/nginx.pid
   # config: /usr/local/nginx/conf/nginx.conf
   nginxd=/home/root/nginx/nginx
   nginx_config=/home/root/nginx/conf/nginx.conf
   nginx_pid=/home/root/nginx/nginx.pid
   RETVAL=0
   prog="nginx"
   # Source function library.
   . /etc/rc.d/init.d/functions
   # Source networking configuration.
   . /etc/sysconfig/network
   # Check that networking is up.
   [ ${NETWORKING} = "no" ] && exit 0
   [ -x $nginxd ] || exit 0
   # Start nginx daemons functions.
   start() {
   if [ -e $nginx_pid ];then
      echo "nginx already running...."
      exit 1
   fi
      echo -n $"Starting $prog: "
      daemon $nginxd -c ${nginx_config}
      RETVAL=$?
      echo
      [ $RETVAL = 0 ] && touch /var/lock/subsys/nginx
      return $RETVAL
   }
   # Stop nginx daemons functions.
   stop() {
           echo -n $"Stopping $prog: "
           killproc $nginxd
           RETVAL=$?
           echo
           [ $RETVAL = 0 ] && rm -f /var/lock/subsys/nginx /home/root/nginx/nginx.pid
   }
   # reload nginx service functions.
   reload() {
       echo -n $"Reloading $prog: "
       #kill -HUP `cat ${nginx_pid}`
       killproc $nginxd -HUP
       RETVAL=$?
       echo
   }
   # See how we were called.
   case "$1" in
   start)
           start
           ;;
   stop)
           stop
           ;;
   reload)
           reload
           ;;
   restart)
           stop
           start
           ;;
   status)
           status $prog
           RETVAL=$?
           ;;
   *)
           echo $"Usage: $prog {start|stop|restart|reload|status|help}"
           exit 1
   esac
   exit $RETVAL
   ```

   ```
   chmod +x /etc/rc.d/rc.local
   echo '/etc/init.d/nginx start' >> /etc/rc.local
   ```

   ​使用OpenSSL生成证书(建议不要密码)

   1、生成RSA密钥的方法

   `openssl genrsa -des3 -out privkey.pem 2048`

   这个命令会生成一个2048位的密钥，同时有一个des3方法加密的密码，如果你不想要每次都输入密码，可以改成：

   `openssl genrsa -out privkey.pem 2048`

   建议用2048位密钥，少于此可能会不安全或很快将不安全。

   2、生成一个证书请求
   `openssl req -new -key privkey.pem -out cert.csr`
   这个命令将会生成一个证书请求，当然，用到了前面生成的密钥privkey.pem文件 这里将生成一个新的文件cert.csr，即一个证书请求文件，你可以拿着这个文件去数字证书颁发机构（即CA）申请一个数字证书。CA会给你一个新的文件cacert.pem，那才是你的数字证书。

   如果是自己做测试，那么证书的申请机构和颁发机构都是自己。就可以用下面这个命令来生成证书：
   `openssl req -new -x509 -key privkey.pem -out cacert.pem -days 1095`

   这个命令将用上面生成的密钥privkey.pem生成一个数字证书cacert.pem

   配置nginx

   ```
   server
   {
   listen 443;
   ssl on;
   ssl_certificate /var/www/sslkey/cacert.pem;
   ssl_certificate_key /var/www/sslkey/privkey.pem;
   server_name 192.168.1.1;
   index index.html index.htm index.php;
   root /var/www/test;
   }
   ```

   ​
