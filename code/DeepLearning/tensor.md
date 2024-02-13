### Shape

> 猜猜我是谁？百变怪！

张量有形状的概念

- shape：表示该张量具有几个维度，或者说张量的阶级
- shape = 0：纯量（标量）
  - tf.constant(2)
- shape = 1：向量
  - tf.constant([2, 2])
    - 和这个写法没有区别`tf.constant((2, 2))`
- shape = 2：矩阵
  - tf.constant([[1, 2, 3], [4, 5, 6]])
- ...

想要获得某个`tensor`的形状直接`.shape`即可，或者`.get_shape`

#### Code

```python
>>> a = tf.constant(shape=[2, 2])
>>> a.shape
TensorShape([Dimension(2)])

>>> b = tf.constant(3)
>>> b.shape
TensorShape([])

>>> c = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> c.shape
TensorShape([Dimension(2), Dimension(3)])
```

<br>

### Create

> tensor 是tensorflow的基本数据类型

#### Constant

顾名思义，`constant`是常量的意思，它被定义后其值不被允许改变

tensorflow允许多种类型`dtype`的常量存在int32、int64、float32、bool、string等

```python
# 实际上dtype可以省略不写，别忘了这是python程序会自动识别
a = tf.constant(40, dtype=tf.int32)
b = tf.constant(40.50, dtype=tf.float32)
c = tf.constant('hello world', dtype=tf.string)
```

#### Zero

- 用零填充该形状的张量

- 注意所有由TF默认生成的 `zeros` & `ones` 都是float类型
- 需要指定`shape`！！！

```python
# 虽然都是零但也分是dtype的
afloat = tf.zeros([2, 2])
aint = tf.zeros([2, 2], dtype=tf.int32)
'''
afloat:
[[0. 0.]
 [0. 0.]]

aint:
[[0 0]
 [0 0]]
'''

# 也存在不用指定shape的情况，会继承afloat的dtype
b = tf.zeros_like(afloat)
'''
[[0. 0.]
 [0. 0.]]
'''
```

#### One

- 换成用一填充

```python
c = tf.ones([2, 2])
d = tf.ones_like(c)
'''
两个长一样的
[[1. 1.]
 [1. 1.]]
'''
```

#### Fill

- 可以自己选择用什么值来填充

```python
e = tf.fill([2, 2], value=2)
'''
[[2 2]
 [2 2]]
'''
```

#### Placeholder

- 一定要定义dtype

```python
# 定义占位符
hold = tf.placeholder(dtype=tf.int32)

# 在会话中给hold赋值
with tf.Session() as sess:
    hold = sess.run(hold, feed_dict{hold: 3})
    # 3
```

#### Variable

- 变量，可在程序运行中改变其值
- 要注意到上述所有创建方式创建的都是常量

- 变量一定要被显示初始化才可运行

```python
# 定义变量
v = tf.Variable(30)
# 初始化所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 需要先跑一边初始化变量的初始化器
	sess.run(init)
    # 变量才能用
	print(v.eval())
```

<br>

### Change

> I am No One

- 静态改变

  - 仅能填充`placehold`的`None`

  - ```python
    set_shape()
    ```

- 动态改变

  - 改变维度但总数不能变

  - 也就是改变后的维度相乘积应该与原本一致

  - ```python
    reshape()
    ```

#### Code

```python
# 静态形状
a_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
a_p.set_shape([2, 3])

# 动态形状
c_p = tf.constant([[1, 2, 3], [4, 5, 6]])
c_p = tf.reshape(c_p, [3, 2])
```

<br>

### TensorBoard

> 所见即所得

TensorFlow提供的可视化学习的工具，便于对程序的理解、调试与优化

#### Stage

- **构建图阶段**
  - 为每个想要观测到的变量起个可供阅读的命名空间
  - 收集每个需要观察的变量
  - 合并上述收集到的变量
- **执行图阶段**
  - 创建事件文件
    - 可在事件文件下开启命令行查看可视化网络
    - 注意logdir参数传的是文件夹名称文件夹下存的是`event`文件
    - `tensorboard --logdir=summary --host=127.0.0.1`
  - 更新迭代的变量
  - 将迭代后的变量写入事件文件

#### Scope

- Why
  - 命名空间可以帮助我们在查看`tensorboard`时更直观观察到每个变量
  - 使代码结构更清晰

```python
with tf.variable_scope('a'):
    a = tf.Variable(30)
```

#### Summary

**收集变量**

- 参数：`name='', tensor`
- scalar：单值变量（纯量）
- histogram：高纬度的变量（非纯量）

```python
tf.summary.scalar('error', error)
tf.summary.histogram('weights', weights)
tf.summary.histogram('bias', bias)
```

**合并变量**

- 一键合并所有变量

- ```python
  merged = tf.summary.merge_all()
  ```

**事务文件**

- 创建事务文件

- ```python
  file_writer = tf.summary.FileWriter(logdir='', graph=sess.graph)
  ```

- 更新事务文件

  - 我们每次跑一遍图都会修改一些变量，如果我们想要在tb中观察到这些变量的变化我们需要每跑一遍就进行整合一次

  - ```python
    # 每次迭代运行一次合并变量操作
    summary = sess.run(merged)
    # 每次迭代后的变量写入事件文件
    # 参数：summary, golbal_step
    file_writer.add_summary(summary, i)
    ```

#### Code

```python
# 命名空间
with tf.variable_scope('myScope'):
    myScope = tf.Variable(30)
    
# 收集变量
tf.summary.scalar('myScope', myScope)

# 合并变量
merged = tf.summary.merge_all()
    
# 初始化变量
init = tf.global_variables_initializer()

# 开启会话
with tf.Session() as sess:
    sess.run(init)
    
    # 创建事务文件
    file_writer = tf.summary.FileWriter('../board/linear', graph=sess.graph)
    
    # 跑图
    for i in range(10):
        # 合并变量
        summary = sess.run(merged)
        
        # 更新事务文件
        file.writer.add_summary(summary, i)
```

<br>

### Flags

> 小黑框它不帅吗？

tensorflow提供了给程序传参的功能

有很多人工智能算法需要人为的调整超参来训练出更好的模型，而我们可以在程序启动之前使用命令行参数输入我们想要的超参

#### 定义参数

- `tf.app.flags`
  - .DEFINE_string(flag_name, default_value, docstring)
  - .DEFINE_integer(flag_name, default_value, docstring)
  - .DEFINE_boolean(flag_name, default_value, docstring)
  - .DEFINE_float(flag_name, default_value, docstring)

```python
tf.app.flags.DEFINE_integer('argv', 50, '参数')
```

#### 获取命令行参数

```python
# 可以使用FLAGS获取参数
FLAGS = tf.app.flags.FLAGS
```

#### 使用参数

```python
tf.app.flags.DEFINE_integer('argv', 50, '参数')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    print(FALGS.argv)
    
if __name__ == '__main__':
    tf.app.run()
```

在命令行使用`python flags.py --argv 30`

<br>

### Saver

> 模型训练这么久当然要保存下来啊！

tensorflow还提供了一个保存模型的api

恰如`joblib`但远比其高级多了，*也复杂多了*

#### Save

默认最多同时存放 5 个模型
`save()`会保存四个文件在目标地址

- .index 和 .data 文件记录了所有变量的取值
- .meta 记录了TF的图结构
- checkpoint文件记录了目录下所有模型文件列表

```python
# 创建一个saver对象
saver = tf.train.Saver()

# 需要在会话中保存
with tf.Session() as sess:
    saver.save(sess, './tmp/tmp.ckpt')
```

#### Restore

restore会恢复之前保存的变量
但在恢复之前最好确认是否有文件可以恢复，使用os检查`chekpoint`即可

```python
import os

# 创建一个saver对象
saver = tf.train.Saver()

# 同样需要在会话中恢复
with tf.Session() as sess:
    # 检查是否存在checkpoint
    if os.path.exists('./tmp/checkpoint'):
        saver.restore(sess, './tmp/tmp.ckpt')
```

<br>

### Demo-Linear

> 深入感受一下深度学习的魅力

```python
import tensorflow as tf
import os

# 接收参数
tf.app.flags.DEFINE_float('rate', 0.1, '梯度下降的学习率')
# 可以使用
FALGS = tf.app.flags.FLAGS


def linear(flags):
    print(flags)
    # prepare data
    with tf.variable_scope('data'):
        '''
        可能用到的计算
        tf.matmul(x, y):相乘
        tf.square(x):平方
        tf.reduce_mean(x):均值
        '''
        # 搞一百个随机数据，将这些数据作为训练集x_train
        x = tf.random_normal(shape=[100, 1])
        # 给上述的每一个样本准备标签y_train
        # y = 0.8 * x + 0.7
        # 注意x的shape=[100, 1]矩阵相乘只能乘[1, ?]的矩阵
        y_train = tf.matmul(x, [[0.8]]) + 19.7

    # model
    with tf.variable_scope('model'):
        # 构建变量weights和bias
        # 特别注意我们需要的是一个一行一列的矩阵
        w = tf.Variable(tf.random_normal(shape=[1, 1]))
        b = tf.Variable(tf.random_normal(shape=[1, 1]))
        # 需要注意这里x和w的顺序不能错，否则无法相乘
        y_predict = tf.matmul(x, w) + b

    # loss
    with tf.variable_scope('loss'):
        # 均方误差：方差求平均
        loss = tf.reduce_mean(tf.square(y_train - y_predict))

    # optimizer
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=flags.rate).minimize(loss)

    # summary
    tf.summary.histogram('w', w)
    tf.summary.histogram('b', b)
    tf.summary.scalar('loss', loss)

    # merge
    merged = tf.summary.merge_all()

    # save
    saver = tf.train.Saver()

    # initial
    init = tf.global_variables_initializer()

    # session
    with tf.Session() as sess:
        sess.run(init)

        # tensorboard
        board = tf.summary.FileWriter('../board/linear', graph=sess.graph)

        # training
        print('before train:\nw:%.2f, b:%.2f, loss:%.2f' % (w.eval(), b.eval(), loss.eval()))
        for i in range(500):
            sess.run(optimizer)

            # 更新tb
            summary = sess.run(merged)
            board.add_summary(summary, i)

            # 每十次训练更新一下保存的模型
            if not (i % 10):
                saver.save(sess, '../models/linear/linear.ckpt')

            print('第%d次训练，损失值为%f' % (i, loss.eval()))
        print('after train:\nw:%.2f, b:%.2f, loss:%.2f' % (w.eval(), b.eval(), loss.eval()))

        if os.path.exists('../models/linear/checkpoint'):
            saver.restore(sess, '../models/linear/linear.ckpt')


def main(argv):
    linear(flags=FALGS)


if __name__ == '__main__':
    tf.app.run()

```

