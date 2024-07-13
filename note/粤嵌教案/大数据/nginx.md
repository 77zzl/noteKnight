### nginx

***



#### 作用

***

nginx监听主目录下html文件中的日志信息，并将信息存储到logs中



#### 安装及配置

***

```
# 下载所需插件
yum -y install zlib zlib-devel gcc gcc-c++ libtool openssl-devel autoconf automake libtool pcre*

# 解压到/export/servers/下
tar -zxf nginx-1.15.8.tar.gz -C /export/servers

# 重命名
mv nginx-1.15.8 nginx

# 配置安装目录，进入nginx目录下
./configure --prefix=/home/hadoop/nginx

# 编译及安装
make && make install

# 修改配置文件，取消log_format和access_log的注释，修改第一行为root
vim conf/nginx.conf
```



#### 启动并使用

***

**启动**，进入nginx/sbin目录下

```
# 重新启动为 ./nginx -s reload
./nginx
```



**使用**

```
# 进入html目录下的页面，并刷新
tail -F /home/hadoop/nginx/logs/access.log
```

