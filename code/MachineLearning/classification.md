### 划分数据集

> 有作业也应该有考试

在监督学习下，我们需要将数据集分成两部分

- 训练集：用于训练模型
- 测试集：用于检测模型

```python
from sklearn.model_selection import train_test_split

# 随机种子是个超参数，接收两个参数，一个是数据集一个是特征名称
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
```

<br>

### 转换器

> 我们初始化的转化器是个啥？

- fit和transform没有任何关系，是数据处理的两个不同环节，fit_transform仅仅是为了编码方便

- sklearn里的封装好的各种算法使用前都要fit，fit是相对于整个代码而言的，为后续API服务。fit之后，才可以调用各种API方法，transform只是其中一个API方法，所以当你调用transform之外的方法，必须要先fit。

- fit原义指的是安装、使适合的意思，其实有点train的含义，但是和train不同的是，它并不是一个训练的过程，而是一个适配的过程

##### `fit`

> 既然所有api都依赖于先fit一遍，所以fit干了什么？

简单来说，就是求得数据集的均值、方差、最大值、最小值，这些数据集固有的属性。

##### `transform`

按照调用的api在fit的基础上进行操作，如归一化、标准化、降维等

##### `fit_transform`

```python
# 这里要注意一点！
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
sc.fit_tranform(X_train)
sc.tranform(X_test)
```

**我们需要保持训练集和测试集的标准一致，即测试集用的fit()应该与训练集用的一样！**

以上内容参考自：九点澡堂子

原文链接：https://blog.csdn.net/weixin_38278334/article/details/82971752

<br>

### 估计器

> 机器学习的算法从此开始

`sklearn`中提供了一类已经实现了算法的api，我们正是利用这些估计器来帮助我们完成模型训练的工作

- classification
  - neighbors k-近邻
  - MultinomialNB 贝叶斯
  - LogisticRegression 逻辑回归
- linear
  - LinearRegression 线性回归
  - Ridge 岭回归
- 无监督学习
  - KMeans 聚类

<br>

### Stage

> 机器学习全攻略就此开始！

1. 获取数据`sklearn.datasets`或者`panda.read_`
2. 数据抽取`Vectorizer`
3. 特征预处理 + 特征降维（注意这一步不是必须的，视需求而定）
4. 初始化一个估计器并放入测试集和标签进行训练
5. 进行模型评估

<br>

### K-近邻(KNN)

> 物以类聚，人以群分

##### Algorithm

根据数据间的距离将邻近的数据归为一类，距离使用欧氏距离

*欧式距离：两点间的直线距离*

##### Code

```python
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
# 在knn中标准化是很必要的因为需要用到欧式距离，标准化能够使得变量之间起相同的作用
transfer = StandardScaler()
'''
复习一下，数据预处理是作用在训练集和测试集上的
但是为了保证标准一致，测试集应该使用训练集的fit()
'''
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

'''
进行模型训练，初始化一个估计器
K-近邻接收两个超参数
n_neighbors：默认为5，表示查询默认使用的邻居数
algorithm：默认为'auto'，可选用于计算最近邻居的算法‘auto’，‘ball_tree’，‘kd_tree’，‘brute’
'''
estimator = KNeighborsClassifier(n_neighbors=9)
# 参数为训练集和训练集对应标签
estimator.fit(x_train, y_train)

'''
模型评估
sklearn提供两种评估方法
对比真实值和预测试：predict
计算准确率（所有预测值中为真的概率）：score
'''
# predict将直接返回按照模型预测出的结果，需要手动对比
predict = estimator.predict(x_test)
print(predict == y_test)
# score可以直接返回准确率
score = estimator.score(x_test, y_test)
print(score)
```

n_neighbors的取值至关重要

- 取值过小：容易受异常点影响独立成为一个类别
- 取值过大：容易受样本不均影响，导致某个重要但数据不多的类别被其他类别吞并

##### Summarize

- 优点：
  - 简单，易于理解，易于实现，无需训练

- 缺点：
  - 懒惰算法，对测试样本分类时的计算量大，内存开销大
  - 必须指定K值，K值选择不当则分类精度不能保证

- 使用场景：
  - 小数据场景，几千～几万样本，具体场景具体业务去测试

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# KNN必须进行标准化
transfer = StandardScaler()

# 估计器
estimator = KNeighborsClassifier(n_neighbors=9)

# 评估模型
# 两个测试集、标签
print(estimator.score(x_test, y_test))
```

<br>

### 超参数搜索

> 大名鼎鼎的CV

##### Algorithm

超参数的调节往往复杂，我们一般选择交叉验证来评估哪个超参数更合适，我们将测试集均分，并循环选择其中一份作为验证集，来判断最优超参数

**交叉验证法**

- k折交叉验证：将一个数据集划分为`k`等份，并选取其中一份作为验证集，剩下的`k - 1`份作为训练集，分别求出每个模型的验证集误差，选择误差最小的模型
- CV误差：`k`个模型的验证集误差均值
- k值是个超参数由人为设定，k值的选取将造成三个方面的影响
  - 训练模型的偏差：k值越小，原始数据和训练数据**分布差距**越大，偏差越大
  - 训练模型的方差：k值越大，模型的**相关性**越大，而由于验证集只有一个，因此验证结果波动较大，方差也大
  - 计算消耗：k值越小**需要计算**的次数越少
- k值的选取方案：
  - 数据量小：10折
  - 数据量大：5折
- 交叉验证时一般选择按照类别比例来划分训练集和验证集，避免因为样本类别不均衡导致的结果失真，这种按照类别比例划分的方法叫做：**分层交叉验证**

##### Code

```python
def CV():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 到此之前都是常规操作，记得此时估计器无需填写需要交叉验证的参数
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 选择需要验证对比的估计器参数
    param_dict = {"algorithm": ['ball_tree', 'kd_tree', 'brute']}
    # 进行交叉验证
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)
    # 既然都用交叉验证了这个没啥用
    print('准确率:', estimator.score(x_test, y_test))
    # 可以无脑估计一下结果
    print('交叉验证中的最佳结果:', estimator.best_score_)
    # 好东西
    print('最好的参数模型:', estimator.best_estimator_)
    # useless
    print('每次验证后的准确率结果:', estimator.cv_results_)
```

##### Summarize

```python
from sklearn.model_selection import GridSearchCV


# 需要进行交叉验证的模型
estimator = KNeighborsClassifier()

# 需要进行选择的超参
param = {'n_neighbors': [3, 5, 7, 9]}

# 初始化交叉验证估计器
estimator = GridSearchCV(estimator, param, cv=4)
```

<br>

### 朴素贝叶斯分类(NB)

> 理想状态下的美好世界

##### Algorithm

在了解这个分类之前我们需要先了解这个分类思想

我们的目的是根据特征找到两个特征向量是否属于同一类别，而恰巧贝叶斯公式就解决了这个问题，能够为我们找到两个事件的关联程度

现在我们来看看朴素贝叶斯分类的定义：`在假设所有特征都互相独立的条件下，各个事件的关联概率`，我们根据这个概率给每个特征向量分类完成需求

**但我们需要注意两点**

- 为了避免计算出某个概率为零，我们需要拉普拉斯平滑系数来解决这个问题
- 理论上朴素贝叶斯的计算结果是完全正确的，但因为现实中很多事件并非完全独立的导致概率偏差

##### Code

```python
def nb():
    news = fetch_20newsgroups(data_home='../data/', subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3, random_state=22)
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 可以看到除了这里使用朴素贝叶斯估计器外代码和KNN没有任何区别
    estimator = MultinomialNB()
    param = {'alpha': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 2.75]}
    estimator = GridSearchCV(estimator=estimator, param_grid=param, cv=4)
    estimator.fit(x_train, y_train)
    score = estimator.score(x_test, y_test)
    print(score)
    print('best score:', '\n', estimator.best_score_)
    print(estimator.best_estimator_)
```

```python
# 虽然很不好，但我想压行
news = fetch_20newsgroups(data_home='../data/', subset='all')
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3, random_state=22)
transfer = TfidfVectorizer()
x_train, x_test = transfer.fit_transform(x_train), transfer.transform(x_test)
param = {'alpha': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 2.75]}
estimator = GridSearchCV(estimator=MultinomialNB(), param_grid=param, cv=4)
estimator.fit(x_train, y_train)
# 验证数据
print(estimator.score(x_test, y_test))
print('best score:', '\n', estimator.best_score_)
print(estimator.best_estimator_)
```

##### Summarize

- 优点
  - 发源于古典数学理论，具有稳定的分类效率
  - 对缺失数据不敏感
  - 分类准确度高，速度快
- 缺点
  - 如果不满独特征相互独立时效果不佳
- 无需标准化
  - 所有“概率”都不需要标准化，因为概率与取值无关只关心变量的分布和变量之间的条件概念

```python
from sklearn.naive_bayes import MultinomialNB


# 估计器
estimator = MultinomialNB()

# 评估模型
# 两个测试集、标签
print(estimator.score(x_test, y_test))
```

<br>

### 决策树

> 解开人工智能的神秘面纱竟是 if-else

##### Algorithm

有一堆信息熵和信息增益的公式但我没看懂

但有两个很有用的概念：

- 树状图可视化工具：`dot`
- 随机森林：分类算法中具有极好的准确率
  - 包含多个决策树，并采用bootstrap随机抽样模型
  - 每个决策树的工作原理，一共有N个样本，M个特征
    - 从所有样本中随机抽取一个样本重复N次
    - 随机选出m个特征(m <= M)
    - 进行训练
  - 集成学习方法：同时进行多个模型学习，并通过投票选择最优模型
  - bootstrap抽样
    - 随机抽取：每个决策树训练出来的总不能一样吧
    - 有放回抽样：如果是无放回的会导致决策树是片面的
    - 从数据集中抽取几个样本标记后再放回，并再次抽取统计有多少是上一次抽取的

##### Code

```python
import panda as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


# 用一个泰坦尼克号生还预测的案例来学习
def titan():
    data = pd.read_csv('../data/titan/train.csv')
    # 数据太大了缩小一点
    data = data.query('18<Age<55')
    # 该数据集中包含了标签，需要把可用特征和标签提取出来
    x, y = data[['Pclass', 'Age', 'Sex']], data['Survived']
    # 处理缺失值，将年龄为空的样本用平均年龄代替
    x['Age'].fillna(x['Age'].mean(), inplace=True)
    # 数据提取
    transfer = DictVectorizer(sparse=False)
    x = transfer.fit_transform(x.to_dict(orient='records'))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    '''
    CV交叉验证选超参
    注意很有意思的一点，如果用了CV验证则不能使用树状图生成了！
    estimator = DecisionTreeClassifier()
    param = {'max_depth': [3, 5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator=estimator, param_grid=param, cv=4)
    estimator.fit(x_train, y_train)
    print('最佳模型：', estimator.best_estimator_)
    print('最佳分数：', estimator.best_score_)
    '''
    '''
    决策树可视化，可以先用cv找出最佳超参数再用可视化导出图像，虽然不知道图像有什么用
    estimator = DecisionTreeClassifier(max_depth=3)
    estimator.fit(x_train, y_train)
    export_graphviz(estimator, out_file='../data/titan_tree.dot', feature_names=transfer.get_feature_names())
    
    # 在文件生成的目录下用命令行跑，可生成可视化png图像
    # dot -Tpng titan_tree.dot -o titan_tree.png
    '''
    # 随机森林
    estimator = RandomForestClassifier()
    param = {'n_estimators': [120, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param, cv=4)
    estimator.fit(x_train, y_train)
    print(estimator.score(x_test, y_test))
    print(estimator.best_score_)
    print(estimator.best_estimator_)
    # max_depth=5, n_estimators=120, min_samples_leaf=1, min_samples_split=2,
```

##### Summarize

- 决策树和随机森林都是通过信息增益进行决策，不关系变量的具体取值因此无需进行标准化

- 决策树
  - 优点：
    - 简单理解(if-else)，具有可视化工具属于白盒模型
  - 缺点：
    - 容易训练出过于复杂的树，导致出现过拟合现象
    - 无法保证准确性
- 决策森林
  - 优点：
    - 具有极好的准确率
    - 可以运行在大数据上，且因为bootstrap抽取无需降维处理
    - 能够估计各个特征再分类问题上的重要性
  - 缺点：
    - 决策森林具有不可解释性，属于黑盒模型

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


# 决策树
# 我个人认为决策树的最好做法是先CV出最优模型再进行可视化
estimator = DecisionTreeClassifier()
param = {'max_depth': [3, 5, 8, 15, 25, 30]}
estimator = GridSearchCV(estimator=estimator, param_grid=param, cv=4)
estimator.fit(x_train, y_train)
estimator = estimator.best_estimator_

# 可视化
estimator.fit(x_train, y_train)
export_graphviz(estimator, out_file='../data/titan_tree.dot', feature_names=transfer.get_feature_names())

# 随机森林
estimator = RandomForestClassifier()
```



