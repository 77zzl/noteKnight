### Feature

> 只要是人工智能就永远离不开特征这个主题

虽然深度学习帮我们免去了特征预处理的种种，但显然特征提取还是我们程序员的活

#### Stage

- 创建文件名队列，用于后续操作
- 将文件内容解码为tensorflow方便处理的类型
- 设置`tensor`的`shape`
- 批处理队列里的内容
- 在会话中开启线程按照上述步骤处理队列里的内容

<br>

### Reader

> 喜闻乐见生产者

我们想要读取的文件往往特别的大，一次性读完显然是不实际的，因此tf提供了利用线程读取队列内容的功能，使得我们可以并发地读取大数据

- **首先**我们将需要读取的数据地址放进文件名队列中

  - ```python
    file_queue = tf.train.string_input_producer(filename_list)
    ```

- **再**根据读取的数据不同使用不同的读取器

  - 创建一个读取器对象

    - ```python
      reader = tf.TextLineReader()
      ```

  - 文本：`tf.TextLineReader`

    - 每次只读取一行

  - 图片：`tf.WholeFileReader`

    - 每次读取整张图片

  - 二进制：`tf.FixedLengthRecordReader()`

    - 这个读取器直译叫做定长记录，即每次读取规定长度的数据（字节）
    - 该方法必须传入参数`record_bytes`指定记录的长度

  - TFRecords文件：`tf.python_io.TFRecordReader()`

  - **需要注意reader每次只能读取一个文件，因此需要批处理来处理所有我们需要的文件**

- **最后**从读取器中拿出待处理的数据

  - ```python
    key, value = reader.read(file_queue)
    ```

  - 注意此时返回的是一个元组类型，内包含一对`key:value`均是tensor类型

    - key：文件名队列的值
    - value：按照文件名队列读取的内容，此时的内容仍是原本的编码格式

### Decode

> 对于不同的数据需要转化成统一的编码方式进行学习

- 将不同类型的文件解码成同一的tensor类型文件`tf.uint8`

  - 文本：`tf.decode_csv()`
  - jpeg图片：`tf.image.decode_jpeg()`
  - png图片：`tf.image.decode_png()`
  - 二进制：`tf.decode_raw()`
    - 该方法必须传入参数`out_type`指定解码的类型

- 注意一个**小细节**，解码之后的tensor形状是未定的，且会自动根据解码类型分配维度

  - ```python
    image:
        Tensor("DecodeJpeg:0", shape=(?, ?, ?), dtype=uint8)
    ```

<br>

### Shape

> 同一大小，同一形状

**为什么要先统一大小呢**？

我们需要将所有张量变成统一的形状才能方便我们进行深度学习，但是我们的数据尺寸不一样，直接`set_shape`会导致出错，所以需要先将尺寸统一再设置静态形状

**图片处理**

- 每个图片有三个维度：图片长度、图片宽度、图片通道数
  - 图片通道数：
    - 对于灰度图片通道数为一，因为我们只需要一个数值来表示颜色深浅即可
    - 对于彩色图片通道数为三，也就是RGE来共同决定像素点色彩
  - 在tensor中对于彩色图片的表示：
    - 单张图片：`[height, width, channel]`
    - 多张图片：`[batch, height, width, channel]`

#### Stage

- 统一尺寸

  - ```python
    # image为原始的图片编码
    image_resized = tf.image.resize_images(image, [?, ?])
    ```

- 设置静态形状

  - ```python
    # 按照不同的[height, width, cannnel]设置静态形状
    image_resized.set_shape([?, ?, ?])
    ```

<br>

### Batch

> 批处理的奥秘：遍历

因为之前的reader只读取了一个文件，所以如果我们想要读取多个文件的话需要批处理操作
而我们之前将文件名都放进队列正是为了现在使用批处理来读取及处理

- batch参数：

  - tensor：需要批处理的内容，可以理解为索引起点
  - batch_size：每次从队列中获取数据的数量，可以理解为索引的位移距离
  - num_threads：队列的线程数
  - capacity：容量，队列中元素的最大数量，这玩意一定要大于等于`batch_size`

- ```python
  image_batch = tf.train.batch([image_resized], batch_size=100, num_thread=1, capacity=100)
  ```

<br>

### Thread

> run Barry ! run

**记得我们之前的读取器吗？**

我们使用读取器从**文件名队列**中读取数据，但读取器每次只能读取一个文件，因此我们需要批处理来帮助我们读取所有文件，而批处理依赖于线程`Thread`来实现。**但**如果单纯的开启线程进行数据读取会导致我们读取到的文件顺序紊乱，因此我们还需要线程协调员`Coordinator`的参与！

#### Stage

- 开启会话

- 开启线程，在线程中加入线程协调员

  - `start_queue_runners`会收集图中的所有队列线程

  - `Coordinator`线程协调员会负责对线程进行管理和协调

  - ```python
    # 注意不能压行！我们需要线程协调员来回收线程
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    ```

- 运行批处理

- 回收线程

  - ```python
    # 发出终止所有线程的命令
    coord.request_stop()
    
    # 线程加入主线程，等待threads结束
    coord.join(thread)
    ```

<br>

### Demo-Dog

> hello puppy!

```python
import tensorflow as tf
import os


def dog():
    # prepare filename list
    filename = os.listdir('../data/dog')
    file_list = [os.path.join('../data/dog', file) for file in filename]

    # prepare filename tf queue
    file_queue = tf.train.string_input_producer(file_list)

    # read
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)

    # decode
    image = tf.image.decode_jpeg(value)

    # change size and shape
    '''
    注意这里不能直接给每个图片都set_shape
    因为每个图片的尺寸都不一样，而给图片set_shape的shape应该严格按照图片的尺寸来set
    所以我们需要将所有图片的尺寸变成一样[200, 200]
    '''
    # size
    image_resized = tf.image.resize_images(image, [200, 200])
    # shape
    image_resized.set_shape(shape=[200, 200, 3])

    # batch
    image_batch = tf.train.batch([image_resized], batch_size=100, num_threads=1, capacity=100)

    # session
    with tf.Session() as sess:
        # 线程协调员和线程
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)

        image_batch = sess.run(image_batch)
        print(image_batch)

        # 回收线程
        coord.request_stop()
        coord.join(thread)


if __name__ == '__main__':
    dog()

```

<br>

### Demo-Cifar

> 学习一下切片与转置！

#### Slice

函数：`tf.slice(inputs, begin, size, name)`
作用：从列表、数组、张量等对象中抽取一部分数据

begin和size是两个多维列表，他们共同决定了要抽取的数据的开始和结束位置

begin表示从inputs的哪个元素开始抽取，直接看坐标判断是哪个元素就好

size表示在inputs的各个维度上抽取的元素个数

- 若begin或size中出现-1,表示抽取对应维度上的所有元素
- 若选择的begin或者size超出范围会报错

```python
>>> a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
>>> a
<tf.Tensor 'Const:0' shape=(2, 2, 3) dtype=int32>
>>> # begin=[0, 1, 1]，从第一个维度第一个元素，第二个维度第二个元素，第三个维度第二个元素开始
... # size=[2, 1, 1]，第一个维度切取两个元素，第二个维度切取一个元素，第三个维度切取一个元素
...
>>> tf.InteractiveSession().run(tf.slice(a, [0, 1, 1], [2, 1, 1]))
array([[[ 5]],

       [[11]]])
>>> tf.InteracitveSession().run(tf.slice(a, [0, 0, 2], [1, 1, 3]))
InvalidArgumentError
```

<br>

#### Transpose

函数：`tf.transpose(a, perm, name)`

作用：将tensor维度按照perm的规则进行转置

比如原本的维度为[2, 3, 4]

perm=[0, 1, 2]
那么原本的维度也不变[2, 3, 4]

perm=[2, 0, 1]
那么相当于让原本第三个维度到第一个维度，第一个维度到第二个维度，第二个维度到第三个维度，也就是[4, 2, 3]

```python
>>> a = tf.random_normal(shape=[2, 3, 4])
>>> a
<tf.Tensor 'random_normal:0' shape=(2, 3, 4) dtype=float32>
>>> tf.transpose(a, [0, 1, 2])
<tf.Tensor 'transpose:0' shape=(2, 3, 4) dtype=float32>
>>> tf.transpose(a, [2, 0, 1])
<tf.Tensor 'transpose_1:0' shape=(4, 2, 3) dtype=float32>
```

**特别注意**，同样是改变维度，但是reshape和transpose完全不同，reshape在改变维度的过程中不会改变扁平化后的数据，而transpose则会按照转置格式进行调整！

```python
>>> tf.InteractiveSession().run(a)
array([[[ 1,  2],
        [ 3,  4]],

       [[ 5,  6],
        [ 7,  8]],

       [[ 9, 10],
        [11, 12]]])
>>> tf.InteractiveSession().run(tf.reshape(a, [2, 2, 3]))
array([[[ 1,  2,  3],
        [ 4,  5,  6]],

       [[ 7,  8,  9],
        [10, 11, 12]]])
>>> # 同样是将shape变成[2, 2, 3]，扁平化后数据顺序改变了
>>> tf.InteractiveSession().run(tf.transpose(a, [1, 2, 0]))
array([[[ 1,  5,  9],
        [ 2,  6, 10]],

       [[ 3,  7, 11],
        [ 4,  8, 12]]])
```

<br>

#### HW

我们了解过图片的表示需要三个维度`height, width, channel`

而在tensor中图片的表示需要按照规定的维度顺序进行排序，必须是`[height, width, channel]`，这种表示方式我们称作**NHWC:batch, height, width, channel**

但现实中的数据往往并不尽如人意，很可能按照`[channel, height, width]`的顺序进行排序，如果我们直接转换成tensor需要的维度顺序`reshape`会出错（产生一段无意义的数据），因此我们需要先reshape成**NCHW**的形式再使用transpose回**NHWC**，来进行维度的调整

#### Code

```python
import tensorflow as tf
import os


class Cifar(object):

    def __init__(self):
        # 初始化操作
        self.height = 32
        self.width = 32
        # 彩色图片的通道数为3
        self.channels = 3

        # 字节数
        # 一条记录包括：标签 + 样本图片
        # 图片3072：长32 * 宽32 * 三色通道数3
        self.image_bytes = self.height * self.width * self.channels
        # 标签1
        self.label_bytes = 1
        # 每个样本总共的字节数
        self.all_bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        # prepare filename queue
        file_name = os.listdir('../data/cifar-10-batches-bin')
        # 构造文件名路径列表
        file_list = [os.path.join('../data/cifar-10-batches-bin', file) for file in file_name if file[-3:] == 'bin']
        # 注意一个细节，file_queue是一个队列，而非tensor对象，只有tensor对象可以sess.run()
        file_queue = tf.train.string_input_producer(file_list)

        # create reader
        # 这个读取器直译叫做定长记录，即每次读取规定长度的数据（字节）
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        key, value = reader.read(file_queue)

        # decode
        # 注意虽然我们所有的解码方式都是将各种类型的数据转化成uint8，但二进制这里需要显示传参
        decoded = tf.decode_raw(value, tf.uint8)

        # 将目标值和特征值切开
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])

        # 调整图片形状
        # 注意本题的图片尺寸是确定下来的32*32因此无需事先resize
        image_reshaped = tf.reshape(image, shape=(self.channels, self.height, self.width))
        print('image_shaped:\n', image_reshaped)
        # 将图片的顺序转为height, width, channels
        # 原来的形状为channels[第0维]*height[第1维]*width[第2维] CHW
        # 目标的形状为height[第1维]*width[第2维]*channels[第0维] HWC
        image_transposed = tf.transpose(image_reshaped, [1, 2, 0])
        print('image_transposed:\n', image_transposed)

        # 调整图片类型
        # 观察数据，image_transposed是整数类型
        image_cast = tf.cast(image_transposed, tf.float32)

        # 批处理
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=1, capacity=100)
        print('label_batch:\n', label_batch)
        print('image_batch:\n', image_batch)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            key_new, value_new, decoded_new, label_new, image_new, image_reshaped_new,image_transposed_new, image_cast_new, label_batch_new, image_batch_new = sess.run([key, value, decoded, label, image, image_reshaped, image_transposed, image_cast, label_batch, image_batch])
            print('key_new:\n', key_new)
            print('value_new:\n', value_new)
            print('decoded_new:\n', decoded_new)
            print('label_new:\n', label_new)
            print('image_new:\n', image_new)
            print('image_reshaped_new:\n', image_reshaped_new)
            print('image_transposed_new:\n', image_transposed_new)
            print('image_cast_new:\n', image_cast_new)
            print('label_batch_new:\n', label_batch_new)
            print('image_batch_new:\n', image_batch_new)

            # 结束回收线程
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    # 实例化Cifar
    cifar = Cifar()
    cifar.read_and_decode()
```

