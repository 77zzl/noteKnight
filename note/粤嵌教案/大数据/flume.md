## flume

***

实时读取服务器本地磁盘的数据，将数据写入到HDFS



### 组成

***

**Source**：收集数据到Agent组件中，Source可处理多种类型日志数据

**Channel**：Source和Sink之间的缓冲区，从而允许Source和Sink运行速率不一，同时保证传输安全；有Memory和File两种类型，Memory可能导致数据丢失

**Sink**：轮询Channel中的事件并批量处理，将这些事件写进HDFS或另一个Agent或其他目的地

**Event**：每个Agent的传输基本单元，包含一个header和byte array



### 优点

***

- Source支持多种采集源
- Sink支持多种目标源
- 当输入速率大于存储速率时，flume将缓冲数据，减小hdfs压力
- 保证数据传输安全可靠



### 拓扑结构

***

1. 串行连接：Sink直接将目标源指向下一个Agent中的Source
2. 单Source，多Channel、Sink：将事件流向多个目的地
3. 负载均衡：一个Source、Channel多个Sink，每个Sink分到一个Agent上解决负载均衡和故障转移
4. Flume Agent聚合（最常见且实用）：每台服务器部署一个flume采集日志，再将日志传输到一个集中收集日志的flume中，经由该Agent上传到HDFS上



### 监听netcat并直接输出到控制台

***

**netcat**

```
nc localhost 44444
[输入发送的内容]
```



flume的使用仅有两步：

1. 编写配置文件
2. 启动flume



**编写配置文件**

Flume提供多种sources以及sink，[点击进入](http://flume.apache.org/releases/content/1.9.0/FlumeUserGuide.html#sources-1)



**开启flume**

```
# 第一种写法
bin/flume-ng agent --conf conf/ --name [name] --conf-file job/flume-netcat-logger.conf -Dflume.root.logger=INFO,consloe

# 第二种写法
bin/flume-ng agent --c conf/ --n a1 -f job/flume-netcat-logger.conf -Dflume.root.logger=INFO,console
```

