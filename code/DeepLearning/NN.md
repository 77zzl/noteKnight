### 神经网络

> ANN大展拳脚

ANN是神经网络的基础，许多高级的神经网络结构(CNN, RNN)最后一层（全连接层FC）便是ANN的内容

ANN也叫多层感知机（MLP），会根据获得的特征参数计算样本的分类概率，并在不断的调参中学习最优模型

我们学过机器学习的模型训练步骤：

- 特征抽取
- 特征预处理  + 特征降维
- 构建模型
- 构建损失函数
- 构建优化函数
- 构建评估函数

而神经网络进行调参的做法叫做**反向传播**，应用复合函数求导的知识，从损失函数开始向前传播，逐步传递至第一层，在此过程中不断调参，使得损失值越来越小

经典的神经网络**NN**包含三层：

- 输入层
- 隐藏层
- 输出层

神经网络的特点：

- 每层都有一到若干个神经元

- 每个神经元之间的连接都有权值
- 同一层次的神经元之间无连接
- 输出结果对应的层次也叫全连接层

### FN

> 使用全连接层来感受一下神经网络的魅力

**神经网络主要用于分类问题！**

类似于逻辑回归使用激活函数`sigmoid`来实现二分类，神经网络使用激活函数`softmax`来进行概率计算，也就是softmax回归

#### Loss & Optimizer

损失函数：

- 交叉熵
  - 用于分类问题
  - 搭配softmax食用更佳
- 均方误差
  - 用于回归问题

优化函数：梯度下降

#### Argmax

tensorflow提供的强大求最大值的方法，可以从不同维度求最大值

函数：`tf.argmax(input, axis)`

将input按照axis的值进行求最大值，axis为0时是最大维度，随着axis增加维度不断减少，如果axis大于最小维度将报错

现在来详细讲一下该函数是如何求最大值的，首先我们要理解成求**每一列**的最大值，axis将决定怎样算一列

```python
>>> a = tf.constant([[[4, 5, 2], [7, 1, 6]],[[1, 0, 2], [9, 2, 7]],[[5, 5, 6],[8
,7, 8]]])
>>> a
<tf.Tensor 'Const:0' shape=(3, 2, 3) dtype=int32>
>>> # 对最大层维度进行求最大值
>>> tf.InteractiveSession().run(tf.argmax(a, 0))
array([[2, 0, 2],
       [1, 2, 2]], dtype=int64)
'''
这里可以把数据看成这样的形式，再对每一列求最大值所在坐标
[[[4, 5, 2], [7, 1, 6]]
 [[1, 0, 2], [9, 2, 7]],
 [[5, 5, 6], [8, 7, 8]]]
可以看到共有六列，并且每一行之间的关系是关于最大维度排序的，在第二个维度有两个值，因此需要将六个最大值每三个分开
'''
>>> # 对第二层维度进行求最大值
>>> tf.InteractiveSession().run(tf.argmax(a, 1))
array([[1, 0, 1],
       [1, 1, 1],
       [1, 1, 1]], dtype=int64)
'''
同样是每列求最大值所在坐标
[4, 5, 2] [1, 0, 2] [5, 5, 6]
[7, 1, 6] [9, 2, 7] [8, 7, 8]
每一行之间的关系都是关于第二维度排序的
'''
```

<br>

### Dome-Mnist-FN

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def full_connection():
    # 1. 准备数据(mnist_data中)
    with tf.variable_scope('data'):
        # 使用tensorflow教程提供的数据接口使用手写数字数据
        mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)
        # 数据和标签都是常量
        # 尺寸28 * 28 = 784
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 2. 构建模型
    with tf.variable_scope('modle'):
        # 权重和偏置都是变量，且初始化为随机值
        Weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]))
        bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
        y_predict = tf.matmul(x, Weights) + bias

    # 3. 构造损失函数
    with tf.variable_scope('error'):
        '''
        激活函数：softmax
        损失函数：交叉熵
        对损失值求平均
        '''
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 优化损失
    with tf.variable_scope('optimizer'):
        # 梯度下降
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 5. 准确率计算
    with tf.variable_scope('accuracy'):
        # a. 比较输出的结果最大值所在的位置和真实值所在的位置是否一致
        equal_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
        # b. 记得类型转换，不然int类型会上取整导致损失部分精度
        # 求平均
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # summary
    tf.summary.histogram('weight', Weights)
    tf.summary.histogram('bias', bias)
    tf.summary.scalar('error', error)
    tf.summary.histogram('accuracy', accuracy)
    merged = tf.summary.merge_all()

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        board = tf.summary.FileWriter('../board/mnist', sess.graph)
        image, label = mnist.train.next_batch(100)
        print('训练之前，损失为：%f' % sess.run(error, feed_dict={x: image, y_true: label}))
        # 开始训练
        for i in range(3000):
            # run完的值千万不能取跟之前一样！！！！
            _, loss, accuracy_value, summary = sess.run([optimizer, error, accuracy, merged], feed_dict={x: image, y_true: label})
            board.add_summary(summary, i)
            print('第%d次训练，损失为：%f，准确率为：%f' % ((i+1), loss, accuracy_value))


if __name__ == '__main__':
    full_connection()
```

<br>

### CNN

> 在全连接层前进行特征提取

**卷积神经网络**

在神经网络中我们有三层结构，分别是输入层、隐藏层和输出层，而为了更好的处理分类任务，我们将隐藏层分为卷积层和池化层，共同实现特征提取的任务

可以理解为我们在全连接层FN前使用CNN进行特征提取的工作

- 卷积层：特征抽取
- 激活层：增加非线性分割能力
- 池化层：降低网络复杂度

#### Convolutional Layer

##### api

实现卷积运算

`tf.nn.conv2d(input, filter, strides=, padding=, name=None)`

- input：输入的图片tensor
- filter：卷积核
- strides：步长
- padding：填充

**通过在原始图像上平移来提取特征**

使用卷积层的原因：

- 特征具有局部性
- 特征可能存在原始图片的任何位置
- 下采样图片不会改变图像目标但可以降低计算量
  - 上采样：放大图片，对图像进行插值
  - 下采样：缩小图片，缩小分辨率

卷积层有三个重要的参数：

- 卷积核
- 步长
- 填充

##### Kernal

每个卷积核都带有若干权重和一个偏置，对特征进行加权运算从而实现特征抽取

每个卷积核参数都是通过反向传播来不断优化的

最外层卷积层可能只能提取显见的特征，而更多层的卷积层能得到更多有价值的特征

##### Stride

步长，用来缩小图片，也就是对图像进行下采样

按照字面意思步长规定了卷积核每次平移的距离，而这个平移距离恰好可以实现对图像像素的缩小

##### Padding

因为我们对图片的处理是使用卷积核平移来进行提取，这会导致原始图片缩小，但有时候我们希望图片的尺寸不发生变化，也就是保留图片边缘信息，此时我们就可以使用填充来解决这个问题

可以使用零填充：在原始图像外填充数值为零的像素，使得图片经过卷积层后尺寸不发生变化

**卷积核的大小一般都是奇数？**

- 更容易padding，使得图片加工后的尺寸为整数

**输出图片的大小？**

- H * W * D
  - (原始图片大小(高度 or 宽度) + 2 * 填充 - 卷积核大小) / 步长
  - D为卷积层层数

- 注意彩色图片是三通道，因此对于每个通道都有一个属于自己的卷积核，最后的输出结果是对三者求和

#### 激活函数

##### Relu

- 有效解决梯度消失的问题
- 计算速度更快
- `tf.nn.relu(features, name)`
  - features：卷积层的结果

#### Pooling Layer

> 去掉不重要的样本，减少参数数量，相当于stride

##### api

`tf.nn.max_pool(value, ksize=, strides=, padding=, name=None)`

- value：tensor的shape[batch, height, width, channels]
- kesize：窗口大小
- strides：步长
- padding：填充方式

有两种池化方法

- 最大池化
- 平均池化

确定一个池化窗口，在这个范围内根据池化方法提取特征值

池化层的计算和卷积层一样

- H * W * D
- （原始大小 - 窗口长度 + 2 * 0）/ 步长 + 1

<br>

#### Dome-Mnist-CNN

见代码mnist_cnn.py

