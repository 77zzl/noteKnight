### K-Means

> 无监督算法登场

##### Algorithm

**注意聚类算法是没有测试集的！**

首先我们需要知道聚类的算法步骤

- 我们先按照超参`n_clusters`随机设置中心点，这些中心点叫做伪中心点（初始聚类中心）
- 遍历所有非聚类中心点，计算它到各个中心点的距离将其与最近的中心点归为一类
- 对所有簇重新求一次中心点（平均值），如果与原中心点一致则结束否则回到上一步

##### Metrics

性能评估：轮廓系数

- 轮廓系数核心在于：低耦合高内聚
- 计算两个值：b_i 与 a_i
  - b_i：到其他簇的最短距离
  - a_i：本簇内的距离平均值
- 我们期望：b_i > a_i
- 轮廓系数的值域在[-1, 1]
- 我们期望的值越接近1越好

##### Code

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score


def kmeans():
    order_products = pd.read_csv("../data/order_products__prior.csv")
    products = pd.read_csv("../data/products.csv")
    orders = pd.read_csv("../data/orders.csv")
    aisles = pd.read_csv("../data/aisles.csv")
    tab1 = pd.merge(orders, order_products, on=["order_id", "order_id"])
    tab2 = pd.merge(tab1, products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, aisles, on=["aisle_id", "aisle_id"])
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])
    table = table[:10000]
    transfer = PCA(n_components=0.95)
    data = transfer.fit_transform(table)
    estimator = KMeans(n_clusters=3)
    '''
    失败的CV搜索超参
    param = {'n_clusters': [3, 4, 5, 6, 7, 8]}
    estimator = GridSearchCV(estimator, param_grid=param, cv=4)
    print('best_score: ', estimator.best_score_)
    print('best_model: ', estimator.best_estimator_)
    '''
    # 使用轮廓系数进行评估，首先要有预测值
    y_predict = estimator.predict(data)
    print('轮廓系数: ', silhouette_score(data, y_predict))
```

##### Summarize

- 优点
  - 直观易懂实用，属于白盒模型
- 缺点
  - 容易收敛到局部最优
  - 每个数据点只能属于一个簇
  - 簇被假定为正圆形

```python
# 导包
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# 聚类一定要无量纲化！保证特征一致才能聚呀
transfer = StandardScaler()

# 估计器及参数
estimator = KMeans(n_clusters = 3)

# 轮廓系数评估模型
y_predict = estimator.predict(data)
score = silhouette_score(data, y_predict)
```

