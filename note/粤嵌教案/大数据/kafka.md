### kafka

***

在开始前明确一点：kafka是一种消息队列

消息队列的作用：解耦



#### kafka的优势

1. 异步处理：提高运行效率
2. 流量削峰：减轻服务器的请求压力



#### kafka的消费模式

1. 一对一：消息被消费后则被删除，一个消息只能被一个消费者消费
2. 发布/订阅：消息被消费后，数据不会删除



#### kafka的架构

![](C:\Users\penny\Desktop\image\kafka架构.png)

- Producer：消息生产者，向Kafka中发布消息的角色。
- Consumer：消息消费者，即从Kafka中拉取消息消费的客户端。
- Consumer Group：消费者组，消费者组则是一组中存在多个消费者，消费者消费Broker中当前Topic的不同分区中的消息，消费者组之间互不影响，所有的消费者都属于某个消费者组，即消费者组是逻辑上的一个订阅者。某一个分区中的消息只能够一个消费者组中的一个消费者所消费
- Broker：经纪人，一台Kafka服务器就是一个Broker，一个集群由多个Broker组成，一个Broker可以容纳多个Topic。
- Topic：主题，可以理解为一个队列，生产者和消费者都是面向一个Topic
- Partition：分区，为了实现扩展性，一个非常大的Topic可以分布到多个Broker上，一个Topic可以分为多个Partition，每个Partition是一个有序的队列(分区有序，不能保证全局有序)
- Replica：副本Replication，为保证集群中某个节点发生故障，节点上Partition数据不丢失，Kafka可以正常的工作，Kafka提供了副本机制，一个Topic的每个分区有若干个副本，一个Leader和多个Follower
- Leader：每个分区多个副本的主角色，生产者发送数据的对象，以及消费者消费数据的对象都是Leader。
- Follower：每个分区多个副本的从角色，实时的从Leader中同步数据，保持和Leader数据的同步，Leader发生故障的时候，某个Follower会成为新的Leader



