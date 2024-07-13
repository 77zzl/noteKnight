# Docker

docker是一个强大的逃课工具，可以帮忙避免程序配置复杂恶心的环境。

其关键在于`镜像`和`容器`两个概念：

镜像：可以把系统某一个时刻的环境定格保存下来，这个保存环境的载体即是镜像

容器：我们使用上述镜像的环境来运行我们自己的项目，也就是让载体跑起来，此时便叫做容器

<br>

## images

```shell
# 列出本地所有镜像
docker images

# 拉取一个镜像
docker pull IMAGE_NAME

# 删除镜像
docker image rm IMAGE_NAME

# 创建某容器的镜像
docker commit CONTAINER_NAME IMAGE_NAME

# 将镜像导出到本地
docker save -o TAR_NAME.tar IMAGE_NAME

# 从本地文件中加载镜像
docker load -i TAR_NAME.tar
```



<br>

## container

```shell
# 查看本地所有容器
docker ps -a

# 利用镜像创建容器
docker create -it IMAGE_NAME

# 启动某容器
docker start CONTAINER_NAME

# 创建并启动某容器
docker run -p 20000:20 --name CONTAINER_NAME -itd IMAGE_NAME

# 停止某容器
docker stop CONTAINER_NAME

# 重启某容器
docker restart CONTAINER_NAME

# 进入容器
docker attach CONTAINER_NAME

# 挂起某容器
Ctrl p + Ctrl q

# 删除容器
docker rm CONTAINER_NAME

# 将容器导出到本地
docker export -o TAR_NAME.tar CONTAINER_NAME

# 将本地文件导入成镜像并命名
docker import TAR_NAME.tar IMAGE_NAME
 
# 重命名容器
docker rename CONTAINER_NAME_OLD CONTAINER_NAME_NEW
```



<br>

## Note

- 为了避免每次使用docker命令都需要sudo，可以将当前用户加入安装中自动创建的docker用户组

```shell
sudo usermod -aG docker $USER
```

- 为了避免导出的文件体积过大，强烈推荐`export`取代`save`

