### TensorFlow

> 定义图，计算图

##### Deep Learning

- 机器学习：特征工程 + 模型训练
- 深度学习：模型训练
  - 需要更多的数据集
  - 需要更多的时间（算力）
  - 可以利用GPU来计算

##### Tensor Flow

- Tensor：张量，TF中的基本操作对象
- Flow：让数据流动起来

TF采用数据流图的形式，每个节点为一个数学操作，线表示张量

TF在程序开始时会默认创建一个图`default_graph`，一般来说一个程序一个默认图就够了

- 图
  - 表示指令之间的依赖关系
  - 在程序构建阶段，所有的数据和操作都会被构建成一张图，此时这张图是静态的
- 会话
  - 运行数据流图的机制
  - 在程序会话阶段，会执行图中的操作

<br>

### Graph

> 定义图

**所有的数据定义、Op操作都在定义图阶段进行**

##### 默认图

```python
# 直接查看tf创建的默认图
graph = tf.get_default_graph()

# 在创建图阶段所有数据都属于默认图
c = tf.constant(10)
print(c.graph)
```



##### 自定义图

```python
# 会发现在自定义中的图与默认图不一样
print(tf.get_default_graph())
with new_graph.as_default():
    a = tf.constant(10)
    print(a.graph)
```



##### Summarize

- 一般情况下默认图就够用了无需自定义新的图

<br>

### Session

> 计算图

**只有在会话中执行图才能让数据“流动起来”**

会话有三种方式开启：

- tf.InteractiveSession
  - 在shell中使用
- tf.Session
  - 传统的会话开启方式
  - 使用后需要关闭
    - sess.close()
- with tf.Session
- session + eval

##### InteractiveSession

```python
# 得到的是一个tensor对象
>>> tf.constant(30)
<tf.Tensor 'Const:0' shape=() dtype=int32>

# 想要让数据按照op执行需要Session
>>> tf.InteractiveSession().run(tf.constant(30))
30
```

##### Session

```python
a = tf.constant(3)
sess = Session()
print(sess.run(a))
# 3
sess.close()
```

##### with Session

```python
a = tf.constant(30)
b = tf.constant(20)
c = tf.add(a, b)
# 开启一段会话，无需手动关闭
with tf.Session() as sess:
    c = sess.run(c)
    # 50
```

##### eval

```python
'''
eval可以得到与run()相同的效果
不过eval只能用于tensor对象，也就是有输出的op
'''
# with Session + eval
a = tf.constant(3)
with Session() as sess:
    print(a.eval())
    # 3
    
# Session + eval
a = tf.constant(3)
sess = tf.Session()
print(a.eval(session=sess))
# 3
```

<br>

### Summarize

可以把`Tensorflow`的步骤理解为静态和动态两个步骤，因为在实际的神经网络中我们要做的肯定不仅仅是定义数据op、让数据流动起来这么简单，但是可以将我们的所有程序块划分为构建阶段和执行阶段

- 在构建阶段我们描绘整个神经网络结构，类似于数据结构

- 在执行阶段我们可能会执行整个神经网络也有可能重复训练整个网络多次，类似于算法







