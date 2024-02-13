### Models

> 我们做的一切都是为了模型

其实就两个点啦

- 模型保存
- 模型读取

```python
from sklearn.externals import joblib


# 加载模型
estimator = joblib.load('../models/test.pkl')

# 读取模型
joblib.dump(estimator, filname='../models/test.pkl')
```

<br>

### Regression

> 梦开始的地方

首先我们要明白，回归模型本质上是决策函数模型，因此模型训练的结果是个函数！

简单的直线函数可以表示为`y = wx + b`的形式

而回归模型做的正是根据数据集不断学习回归系数(权重系数)`w`和偏置`b`的过程 

**优化函数**，用于求解`w`和`b`的方式

- 正规方程
  - 直接求解最优结果，当特征太多时求解速度太慢
- 梯度下降
  - 利用导数不断寻找最低点

**性能评估**

- 均方误差：方差求平均

<br>

### Linear

> 正规方程

##### Algorithm

使用正规方程直接求解回归系数和偏置

##### Loss & Optimization

损失函数：最小二乘法

优化函数：正规方程

##### Metircs

均方误差：方差求平均

##### Code

```python
def linear():
    data = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 均方误差接收两个参数：真实标签与预测值
    error = mean_squared_error(y_test, estimator.predict(x_test))
    print('使用Linear模型的均方误差：', error)
    # 使用Linear模型的均方误差： 26.7101375346038
    # print('正规方程-权重系数为：\n', estimator.coef_)
    # print('正规方程-偏置为：\n', estimator.intercept_)
```

##### Summarize

- 优点
  - 直接求解
- 缺点
  - 无法处理大量的数据
- 使用场景
  - 少量数据

```python
# 普通的线性回归无需标准化
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 估计器及参数
estimator = LinearRegression()

# 模型评估：均方误差
error = mean_squared_error(y_test, estimator.predict(x_test))
```

<br>

### SGD

> 随机梯度下降

##### Algorithm

梯度下降使用迭代的方式不断逼近最低点，但是普通的梯度下降容易陷入鞍点，也就是局部最低点，而随机梯度下降解决了这个问题

##### Loss & Optimization

损失函数：最小二乘法

优化函数：梯度下降

##### Metircs

均方误差：方差求平均

##### Code

```python
def sgd():
    data = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    '''
    SGD使用梯度下降优化因此有多个超参
    loss：损失类型
        "squared_loss"：普通最小二乘法
        'huber'：改进的普通最小二乘法，修正异常值，不要随便调
    learning_rate：学习率填充
    penalty：惩罚项
    alpha：正则化程度
    max_iter：最多迭代次数，默认为5
    '''
    # estimator = SGDRegressor(max_iter=1000, eta0=0.001, alpha=0.001)
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)
    # 均方误差接收两个参数：真实标签与预测值
    error = mean_squared_error(y_test, estimator.predict(x_test))
    print('使用SGD模型的均方误差：', error)
    # 使用SGD模型的均方误差： 25.703309542636315
    # print('梯度下降-权重系数为：\n', estimator.coef_)
    # print('梯度下降-偏置为：\n', estimator.intercept_)
    # print('梯度下降-迭代次数：\n', estimator.n_iter_)
```

##### Summarize

- 优点
  - 高效、易实现
- 缺点
  - 需要调节超参（损失函数、优化函数、学习率）
  - 需要多次迭代
- 使用场景
  - 大数据

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


# 对数据标准化敏感
transfer = StandardScaler()

# 估计器及参数
estimator = SGDRegressor()

# 均方误差进行模型评估
error = mean_squared_error(y_test, estimator.predict(x_test))
```

<br>

### Ridge

> 正则化大展身手的时候到了！

##### Algorithm

因为数据集的原因我们的模型常常容易出现欠拟合或过拟合的问题

- 欠拟合：因为数据量过少导致模型训练不到位

- 过拟合：因为某些特征的异常点过多、数据量少也会过拟合、某些训练的特征与结论无关

**解决方案**，那么我们该如何解决这类问题？

- 欠拟合：增加数据

- 过拟合：针对回归模型引入正则化概念

**正则化**：减少异常特征的过程就叫做正则化过程

- L1正则化：使一些w直接为零，删除该特征的影响
- L2正则化：使一些w接近零，削弱某个特征的影响

**岭回归**，可以理解为使用L2正则化的SGD

##### Loss & Optimization

损失函数：最小二乘法 + L2正则化

优化函数：梯度下降

##### Metircs

均方误差：方差求平均

##### Code

```python
def ridge():
    data = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    '''
    使用平平无奇岭回归
    estimator = Ridge()
    '''
    # 使用高端大气岭回归交叉验证，可以传入多个超参
    estimator = RidgeCV(alphas=[0.1, 1.0, 10.0])
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print('岭回归的均值误差：', error)
    print('最佳参数', estimator.alpha_)
```

##### Summarize

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# 凡是正则化都需要对数据进行标准化预处理
transfer = StandardScaler()

# 平平无奇岭回归
estimator = Ridge()

# 高端大气上档次实现了岭回归的交叉验证，可以选择多个超参进行训练
estimator = RidgeCV(alphas=[0.1, 1.0, 10.0])

# 评估模型
error = mean_squared_error(y_test, estimator.predict(x_test))
```

<br>

### Logistic

> 分类算法派来的卧底

**逻辑回归是用来解决二分类问题的机器学习方法**

##### Algorithm

逻辑回归的输入是一个线性回归的结果，也就是一个决策函数

逻辑回归使用激活函数`sigmoid`（注意激活函数有多种）将输入处理成一个概率值，算法根据设定的阈值将样本进行分类

##### Loss & Optimization

损失函数：对数似然损失

优化函数：梯度下降（GD）

##### Metrics

混淆矩阵

| 真实结果(y_test) / 预测结果(predict) | 正例     | 假例     |
| ------------------------------------ | -------- | -------- |
| **正例**                             | 真正例TP | 伪正例FN |
| **假例**                             | 伪反例FP | 真反例TN |

你管这叫混淆矩阵？

- 混淆矩阵只能用来评估二分类模型！

- 由两个大写英文字母组成
  - 第一个字母表示预测结果的真假
    - 真`T`：True
    - 假`F`：False
  - 第二个字母表示预测的结果
    - 预测为真`P`：Positive
    - 预测为假`N`：Negative

混淆矩阵的评估方式有两种，本例选择第二种

- F1-score
  - 查准率（精确率P）：预测为正的所有样本中真实为正的概率
  - 查全率（召回率R）：真实为正的所有样本中有多少被预测为正
  - 调和平均数（F1-score）：2(P + R) / P * R
  - PR曲线：以查全率为横坐标查准率为纵坐标的曲线
- AUC指标
  - 真正率：查全率
  - 假正率（打扰率）：所有真实为负的样本中被误判为正的概率
  - ROC曲线：以假正率为横坐标真正率为纵坐标的曲线
  - AUC指标：指的是ROC曲线和横坐标围成的梯形面积，该面积越大说明模型越好，即相同假正率下真正率更大，此外，该指标还具有重要的物理意义：随机抽取一对正负样本，正样本得分大于负样本的概率
- PR曲线与ROC曲线
  - ROC能同时兼顾正负样本，而PR更关注正样本
  - AUC越大，正样本的预测结果更靠前
  - 正负样本不均时
    - 更关注正样本：PR
    - 更关注负样本或者两者：AUC

本例中我们更关心能否发现癌症患者，也就是抱着宁可错杀一千也不能放过一个的思想，相比于精确率我们更关系打扰率尽可能低，因此选用AUC指标作为模型评估

##### Code

```python
def logistic():
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_name)
    # 将?替换为nan
    data = data.replace(to_replace='?', value=np.nan)
    # 将包含nan的行删除，删除整个样本
    data = data.dropna()
    x = data[column_name[1:10]]
    y = data[column_name[10]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # print("预测的类别：", estimator.predict(x_test))
    print("预测的准确率:", estimator.score(x_test, y_test))
    '''
    classification_report参数：
    真实的目标值（标签）
    估计器的预测值
    类别对应的数字
    类别对应的名字
    '''
    print("精确率和召回率和调和平均数为：\n", classification_report(y_test, estimator.predict(x_test), labels=[2, 4], target_names=['良性', '恶性']))
    # 这个np.where类似于三目运算符
    y_test = np.where(y_test > 2.5, 1, 0)
    print("AUC指标：", roc_auc_score(y_test, estimator.predict(x_test)))
```

##### Summarize

- 注意逻辑回归只能用于二分类问题
- 逻辑回归使用的优化是梯度下降，因此必须标准化处理

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# 特征预处理
transfer = StandardScaler()

# 模型估计器
estimator = LogisticRegression()

# F1
# 参数：标签、估计器、类别对应的数字、类别对应的名字
classification_report(y_test, estimator, labels=[0, 1], target_names=['False', 'True'])

# AUC
# 参数：标签、预测值
roc_auc_score(y_test, estimator.predict(x_test))
```

