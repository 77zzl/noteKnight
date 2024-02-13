我们需要对特征进行处理，这个过程叫做特征工程，特征工程包括三个方面：

- 数据抽取
- 特征预处理
- 特征降维：

<br>

### 加载数据

> 我们从何处得到数据？

##### `sklearn`

**提供了两种获取数据的方式**

首先需要导包`from sklearn.datasets import *`

- 获取小规模数据：`load_`
  - `from sklearn.datasets import load_iris`之后便可直接使用`iris()`
- 获取大规模数据：`fetch_`

```python
# 小规模数据
from sklearn.datasets import load_iris
iris = load_iris()

# 大规模数据
from sklearn.datasets import fetch_20newsgroups
# data_home表示数据下载目录
# subset可选有'train','test','all'
fetch_20newsgroups(data_home=None,subset=‘train’)
```



##### `panda`

**提供的方法可以获取各种文件中的数据**

导包`import panda as pd`

- 从以逗号分隔的文件中获取数据：`pd.read_csv(PATH)`

<br>

### 数据抽取

> 我们的数据可能来自各种地方

需要了解的特征抽取方式也有三种：

- 将字典类型的数据转化为`numpy`
- 将文本转化为`numpy`
- 将文本转化为`tfidf`类型的`numpy`

现在分别了解这三种数据抽取方式

<br>

##### `DictVectorizer`

```python
# 以下内容给我背过
# 导包,extraction是萃取的意思
from sklearn.feature_extraction import DictVectorizer
''' 
初始化一个转化器
DictVectorizer接收一个参数sparse
该参数为False时表示one-hot独热编码
该参数为True时表示one-of-K编码
后者使用.toarray()可以转化为前者
'''
transfer = DictVectorizer(sparse=True)
data = transefer.fit_transform(data)
# 获取特征名称
print(transfer.get_feature_name())
# 获取数据
# 如果是独热编码直接data可以查看数据
print(data.toarray())
```

<br>

##### `CountVectorizer`

将数据转化为词频矩阵，也叫稀疏矩阵

```python
from sklearn.feature_extraction.text import CountVectorizer

# 喜闻乐见初始化转化器
transfer = CountVectorizer()

# 喜闻乐见转化操作
data = transfer.fit_transform(data)

# 喜闻乐见获取特征名
print(transfer.get_feature_name())

# 喜闻乐见获取数据
print(data.toarray())

# CountVectorizer是针对语言处理的转化器，但中文略有不同，因此介绍jieba
import jieba

# jieba的方法可以按照中文语义将词语分割
jieba.cut(text)

# 给出一个压行技巧,假设有个由中文组成的字符串列表text
data = [' '.join(list(jieba.cut(line))) for line in text]
```

<br>

##### `TfidfVectorizer`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
'''
tf:词频
词语总数为 n， a 出现了 m 次，则 tf 为 m / n
idf:逆向文件频率
单词 a 在 m 份文件中出现，文件总数为 n ，idf 为 lg(n / m)
TfidfVectorizer 相当于 CountVectorizer + TfidfTransformer
'''
transfer = TfidfVectorizer()
# 后面喜闻乐见的内容不写了，tfidf对于文本处理很重要
```

<br>

### 特征预处理

> 让数据变得更容易有意义

特征预处理，共有两种方式

- 归一化：将数据控制到`[0, 1]`之间
- 标准化：将数据控制到均值为`0`，标准差为`1`的范围内

<br>

##### `MinMaxScaler`

归一化对噪音敏感，容易受异常数据干扰

```python
from sklearn.preprocessing import MinMaxScaler


# 注意feature_range()必须接收两个参数，不可省略前一个
# 数据将被控制在0到1之间
transfer = MinMaxScaler(feature_range(0, 1))

data = transfer.fit_transform(data[[需要预处理的特征]])
print(data)
```

<br>

##### `StandardScaler`

```python
from sklearn.preprocessing import StandardScaler


transfer = StandardScaler()
data = transfer.fit_transform(data[[需要预处理的特征]])

# 每一列的特征平均值
print(transfer.mean_)

# 每一列的特征方差
print(transfer.var_)
```

<br>

### 特征降维

> 让**特征**更有意义

特征降维的两种方式：

- 特征选择
  - 过滤式
    - 方差选择法
    - 相关系数法
  - 嵌入式、包裹式
- 主成分分析

##### `VarianceThreshold`

过滤掉低方差特征

```python
from sklearn.feature_selection import VarianceThreshold


# 过滤掉方差小于threshold的特征，默认保留所有非零方差特征
transfer = VarianceThreshold(threshold=1)

# iloc[args1, args2], args1为行索引，args2为列索引
# 从原本的数据中取出所有行，列从1到9的数据
data = transfer.fit_transform(data.iloc[:, 1:10])

# 获取数据
print(data)
```

<br>

##### `pearsonr`

过滤掉系数高的特征

相关系数：皮尔逊相关系数反应了变量之间相关关系密切程度

所以我们要做的就是删掉相关系数接近1或-1的特征

值域在[-1, 1]之间，当取值大于零为正相关，小于零为负相关，等于零是表示无相关关系

- abs(r) < 0.4：低相关度
- abs(r) >= 0.4 and abs(r) < 0.7：显著相关
- abs(r) >= 0.7：高度相关

```python
from scipy.stats import pearsonr


# 比想象中好用
pearsonr(data1, data2)
```

<br>

##### `PCA`

主成分分析

将数据按照主成分分析法减少

PCA(n_components)

- 参数为整数：减少到多少特征

- 参数为小数：保留百分之多少特征

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 主成分分析一定要对数据标准化
transfer = StandardScaler()

# 保留原本百分之九十的特征
transfer = PCA(n_components=0.9)

# 将特征保留至两个
transfer = PCA(n_components=2)
data = transfer.fit_transform(data)
```

<br>

### Demo-PCA

```python
import pandas as pd
from sklearn.decomposition import PCA

# 导入数据
products = pd.read_csv('../data/products.csv')
order_products = pd.read_csv('../data/order_products__prior.csv')
orders = pd.read_csv('../data/orders.csv')
aisles = pd.read_csv('../data/aisles.csv')

'''
将四张表合并在一起
merge(args1, args2, args3)
args1、args2：需要合并的数据
args3：on内写两个数据内需要对应合并的字段
类似于select * from args1 left join args2 on arges3
'''
table1 = pd.merge(orders, order_products, on=['order_id', 'order_id'])
table2 = pd.merge(table1, products, on=['product_id', 'product_id'])
table3 = pd.merge(table2, aisles, on=['aisle_id', 'aisle_id'])

# 列联表（panda实在没搞懂）
table = pd.crosstab(table3['user_id'], table3['aisle'])
print(table.shape)

# 主成分降维
transfer = PCA(n_components=0.95)
data = transfer.fit_transform(table)
print(data.shape)
```

