### 字典

python

```python
c = Counter('hello world')
x = c.keys()
print(x)
# dict_keys(['h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'])

x = c.items()
print(x)
# dict_items([('h', 1), ('e', 1), ('l', 3), ('o', 2), (' ', 1), ('w', 1), ('r', 1), ('d', 1)])
```

c++

```c++
// 用c++实现计数器
unordered_map<string, int> cnt;

// 注意auto后有&表示此为引用，可以在循环内修改其值，如果没有&则为拷贝，不影响原数组数据
for (auto& w : words) cnt[w]++;	
for (auto& [w, v] : cnt) printf("%c %d\n", w, v);
```



### 字符串

c++

```c++
string a;
a.push_back('a');
a.push_back('b');
cout << a;
// ab

cout << a.back();
// b
```



### 栈

c++

```c++
# 判空
st.empty();

# 弹出
st.pop();

# 插入
st.push('a');

# 大小
st.size();

# 取栈顶
st.top();
```



### 记忆化搜索

让系统自动记录所有的搜索结果，替代人为使用数组记录（动规）

比如针对如下动规代码，不用理会代码逻辑，此时用了`dp[i]`来记录某种信息，当找到一个满足条件的`j`时，可以很方便地用`dp[j] < dp[i]`来进行比较，而所有`dp[j]`都是系统曾计算过的，我们使用一个函数来计算`dp`数组的不断迭代，当需要某个曾经的结果时，同样可以直接使用

```python
ss, n = set(dictionary), len(s)
dp = [0] * (n + 1)
for i in range(1, n + 1):
    dp[i] = dp[i - 1] + 1
    for j in range(i):
        if s[j:i] in ss and dp[j] < dp[i]:
            dp[i] = dp[j]
return dp[n]
```

可以使用记忆化搜索写成这样，系统会自动记录`dp()`函数的所有结果，相当于`dp[i]`

```python
d = set(dictionary)
@cache
def dp(i:int) -> int:
    if i < 0:
        return 0
    res = dp(i - 1) + 1
    for j in range(i + 1):
    	if s[j:i + 1] in d:
            res = min(res, dp(j - 1))
        return res
return dp(len(s) - 1)
```





### replace

python

```python
s = 'abcd'
s.replace('a', '').replace('b', 'e')
# 'ecd'
```



### pairwise

```py
from itertools import pairwise

a = [i for i in range(5)]
for x, y in pairwise(a):
    print(x, y)
'''
0 1
1 2
2 3
3 4
'''
```



### ASCII码与字符相互转换

```python
# 字符转ASCII码 ord
# ASCII码转字符 chr

# 用户输入字符
c = input("请输入一个字符: ")
 
# 用户输入ASCII码，并将输入的数字转为整型
a = int(input("请输入一个ASCII码: "))

print( c + " 的ASCII 码为", ord(c))
print( a , " 对应的字符为", chr(a))
```





### 二分搜索

`bisect` 的特点是可以按照值去查找与插入，且函数是用二分实现的

**其中的array必须是升序的有序数组！！！**

```
import bisect
```

查找：`bisect(array, item)` or `bisect_left(array, item)`
插入：`insort(array, item)`

看以下数据：

```python
# bisect & bisect_left
>>> a = [1,4,6,8,12,15,20]
>>> p = bisect.bisect(a, 6)
>>> p
3
>>> p = bisect.bisect_left(a, 6)
>>> p
2

# insort
# insort 也有 insort_left 但没有必要
>>> a = [1,4,6,8,12,15,20]
>>> bisect.insort(a, 13)
>>> a
[1, 4, 6, 8, 12, 13, 15, 20]
```





### list的三种删除方式

- **pop()** 
- **del()** 
- **remove()**

```python
# pop()
list=[11,12,13,14,15]
list.pop()
# [11,12,13,14]

# pop也可传入坐标
list.pop(0)
# [12,13,14]

# del()
list=[11,12,13,14,15]
del(list[1])
# [11, 13, 14, 15]

list=[11,12,13,14,15]
list.remove(11)
# [12, 13, 14, 15]
```



### 绝对值

```python
# abs()
a = -1
abs(a)
# a
```



### 英文字母大小写转换

```python
# upper()
# lower()
```



### 索引序列

```python
# enumerate()
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
for i, c in enumerate(seasons):
    print(i, c, end = " ")
# 0 Spring 1 Summer 2 Fall 3 Winter 

# 学一个骚操作，多复杂的结构遍历都不在话下
cnt = [[1, 'a'], [2, 'b'], [3, 'c']]
for i, (c, ch) in enumerate(cnt):
    print(i, c, ch, end = ' ')
# 0 1 a 1 2 b 2 3 c
```



### max()的参数

- key
- default

```python
# max()
# key
testlist = [9.2, 10, -20.3, -7.0, 9.999, 20.111]
max(testlist, key = abs)
# -20.3
# key的作用相当于对testlist中的每个元素先用key指定的函数进行处理，然后再返回最大值

# default
testlist = [9.2, 10, -20.3, -7.0, 9.999, 20.111]
max(testlist, default = 99)
# 20.111
t = []
max(t, default = 99)
# 99
```



### 字符串查找

```python
# find()
str = 'hello world'
str.find('e')
# 1
str.find('a')
# -1
# 不存在
```



### 字符串反转

- reversed()
- [::-1]

```python
# 不会还有人用这个叭
# reversed
str = 'hello world'
"".join(list(reversed(str)))
# 'dlrow olleh'

# [::-1]
str = 'hello world'
str[::-1]
# 'dlrow olleh'
```

学到一个很帅气的技能，实现翻转列表前n位

```python
a = [1, 4, 2, 5]
a[::-1]
# [5, 2, 4, 1]
a[0::-1]
# [1]
a[1::-1]
# [4, 1]
a[2::-1]
# [2, 4, 1]
a[3::-1]
# [5, 2, 4, 1]

# 可以看出[起始位::从起始位开始向前翻转]
# 当需要返回翻转前n位时：
a[n - 1::-1] + a[n:]
```



### 海象运算符

**python3.8才有**

堪称利器，适用于`if/else` `while` `推导式`

**注意需要用括号框起来**

```python
# :=
# 有点可爱
# 年龄验证
if (age := 19) > 18:
    print("1")

# 密码验证
while (p := input()) != 'password':
    continue

# 反转单词前缀，word原字符，ch需要反转的最后一位
word[:(i := word.find(ch)) + 1][::-1] + word[i + 1:]
```



### 移除开头结尾字符串

```python
# strip()
str = '    hahaha    '
str.strip()
# 'hahaha'

# 只要是出现的参数无论顺序都会被移除
str = '123434321'
str.strip('12')
# '34343'
```



### 计数器Counter

- **Counter()**

```python
# 导包 
from collections import Counter

# 初始化
c = Counter()
c['a'] = 1
c['b'] = 2
c
# Counter({'b': 2, 'a': 1})

s = 'sdflkdsjfd'
c = Counter(s)
c
# Counter({'d': 3, 's': 2, 'f': 2, 'l': 1, 'k': 1, 'j': 1})

# 读取数据 & 更新同理
s = 'hello world'
c = Counter(s)
c['l']
# 3

# 删除
del c['l']
c
# Counter({'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})
c['l']
# 0
```

注意对于未赋值过的数据，Counter将默认赋值为0



#### 获取所有元素

- .elements()

```python
list(c.elements())
# ['h', 'e', 'o', 'o', ' ', 'w', 'r', 'd']
```



#### 获取键值对

- .items()

```python
list(c.items())
# [('h', 1), ('e', 1), ('o', 2), (' ', 1), ('w', 1), ('r', 1), ('d', 1)]
```



#### 按降序查看键值对

- .most_common()

```python
c.most_common()
# [('o', 2), ('h', 1), ('e', 1), (' ', 1), ('w', 1), ('r', 1), ('d', 1)]
```



#### 多个计数器操作

```python
c = Counter(a = 3, b = 1)
d = Counter(a = 1, b = 2)

# 相加
c + d
# Counter({'a': 4, 'b': 3})

# 相减，小于零将自动删去
c - d
# Counter({'a': 2})

# 求最小
c & d
# Counter({'a': 1, 'b': 1})

# 求最大
c | d
# Counter({'a': 3, 'b': 2})
```



### 计数器count

```python
s = 'hello world'
s.count('l')
# 3
```



### 浮点数取整

- math.floor
- math.ceil
- f - int(f)

```python
# math下的函数，上取整与下取整
import math
f = 4.74

# 上取整
math.ceil(f)
# 5

# 下取整
math.floor(f)
# 4

# 在python中，将浮点数强制转换程整数时也是下取整
int(f)
# 4
```



### 浮点数保留小数点后几位

- round()

```python
r = 1.666667
round(r, 2)
# 下取整
# 1.67
```



### 排序

- sort()

- ```python
  # sort 本身没什么好讲的，来学一个逆序排序
  a = [1, 3, 2, 9, 0, 4]
  a.sort()
  # [0, 1, 2, 3, 4, 9]
  a.sort(key = lambda x: -x)
  # [9, 4, 3, 2, 1, 0]
  ```



### 小根堆 & 大根堆

- heapq

```python
# 导包
import heapq

# 初始化
heap = [1, 4, 2, 5]
heapq.heapify(heap)

# 将x压入堆中
heapq.heappush(heap, x)

# 从堆中弹出最小元素
m = heaqp.heappop(heap)

# 返回最小的n个元素
li = heapq.nsmallest(n, heaq)
```

python 的 `heapq`默认为小根堆，转换为大根堆的方式很简单，对于每个存进去的数均乘以`-1`，在数据保证一定为正数的情况下`-heap[0]`便是最大值

```python
# 插入
heapq.heappush(heap, -a)

# 弹出
maxvalue = -heapq.heappop(heap)
```



### 高级字典

- defaultdict()

先聊聊普通字典 `dict()`

```python
d = dict()
d['age'] = 10
d['age']
# 10
d['sex']
# 报错
```

而 `defaultdict()` 的特点就在这里了，但查询的结果不存在时，可以返回一个默认值

```python
# 导包
from collections import defaultdict

# 初始化
d = defaultdict(int)
d['age'] = 10
d['age']
# 10
d['sex']
# 0
```

同理 `defaultdict()` 可创建的默认值还有这些：

```python
d1 = defaultdict(int)
d2 = defaultdict(list)
d3 = defaultdict(str)
d4 = defaultdict(set)
```





### 最大公约数

- gcd()

```python
# gcd() 返回两个值的最大公约数
# 导包
import math

math.gcd(3, 5)
# 1
math.gcd(4, 6)
# 2
```





### 字符串格式化函数

- format()
- f-string

```python
# str.format()
# 用{}和:来代替%

"{} {}".format("hello", "world")
# hello world

# f-string
name = 'zzl'
f'hello {name}'
# 'hello zzl'
```



### 双向队列

- deque

```python
# 导包
from collections import deque

# 初始化
d = deque()

# 添加元素
d.append(1)
d.appendleft(2)
d
# 21

# 拓展列表
d.extend(['a', 'b', 'c', 'e'])

# 清空队列
d.clear()

# 查询
d.index('e')
# 3

# 插入
d.insert(2, 'z')
d
# deque(['a', 'b', 'z', 'c', 'e'])

# 删除
r = d.pop()
l = d.popleft()
c = d.pop('c')

# 反转
d.reverse()
```



### 位运算

- 位运算异或 `^`

```
1 ^ 1 = 0
1 ^ 0 = 1
0 ^ 0 = 0
```

异或可用于算法中 `^ 1` 操作，可将奇数变成偶数，偶数变成奇数

- 位运算与 `&`

```
1 & 1 = 1
1 & 0 = 0
0 & 0 = 0
```

与在算法中可用于 `& 1` 来进行奇偶判断，偶数&1为0，奇数&1为1

**位运算也可用于set**！

```python
>>> e = [1, 2]
>>> a = [2, 4]
>>> set(e) & set(a)
{2}
>>> set(e) | set (a)
{1, 2, 4}
>>> set(e) ^ set(a)
{1, 4}
```

- & ：取交集
- | ：取并集且去重
- ^ ：取并集并把交集去掉

#### 二进制位标记！

- 复习一个知识点，位运算

```
00 | 10 == 10

10 & 11 == 10
```

- 标记第三位

```python
st = 0

st |= 1 << 3

# 在st中标记第n位
st |= 1 << n
```

#### 热知识

```python
# 1，计算整数x的二进制表示有多少个1, 可以消除x最低位的1，while循环计数，直到x=0即可。
while x:
	x &= x - 1

# 2，只保留整数x最低位的1，暨鼎鼎大名的lowbit
x & -x
```





### 数组构建

- 当构建二维数组时要小心两种不同的构造方法效果不一

```python
# *
a = [0] * 2
# [0, 0]
a = [[0, 0] * 2]
# [[0, 0, 0, 0]]

# for _ in range()
a = [0 for _ in range(2)]
# [0, 0]
a = [[0, 0] for _ in range(2)]
# [[0, 0], [0, 0]]
```



### 判断是字母 | 数字

```python
# 字母或数字
str.isalnum()

# 字母
str.isalpha()

# 数字
str.isdigit()

# 字母小写
str.islower()

# 字母大写
str.isupper()
```



### 深度拷贝

- **深拷贝(deepcopy)：** copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象。

```python
import copy

a = [1, 2, 4]
b = a
b[2] = 3
# b == [1, 2, 3], a == [1, 2, 3]
# b的增删改查都将影响a，a把指针赋值给了b

b = copy.deepcopy(a)
b[2] = 4
# b == [1, 2, 4], a == [1, 2, 3]
```



### 生成器

- yield
- yield from

```python
# 看看2022/3/10的每日一题
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        def dfs(node):
            if node:
                yield node.val
                for c in node.children:
                    yield from dfs(c)

        return [v for v in dfs(root)]
```

[589. N 叉树的前序遍历](https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/)

- 首先要认识到 `yield` 的作用
  - 存在yield的函数将会变成一个生成器
  - 生成器的作用可以实现简单的通信，函数会在 `yield` 的地方卡住，等待下一次唤醒
  - 但是要注意生成器需要初试化，初始化后将得到一个生成器对象，该对象通过 `next()` 迭代或者 `for i in run()` 直接使用
  - 使用 `send()` 可以与迭代器内部通信，发送的值将作为函数内 `yield` 的返回值
- `yield from`
  - `yield` 后的值将作为生成器返回值返回给调用函数，而实际上 `yield` 接受另一个迭代器或者生成器作为返回值
  - `yield from generate()` 意味着可以实现生成器的嵌套使用



### 跳出多重循环

- for ... else

```python
# 当最内层break执行后直接跳出所有循环
for i in range(n):
    for j in range(m):
        for k in range(h):
            if check(i, j, k):
                break
        else:
            continue
        break
    else:
        continue
    break
```



### 按照值返回索引

- index

```python
# python无所不能
a = [1, 3, 4, 9, 1]
a.index(1)
# 0
```



### 返回十进制数的二进制

```python
bin(8)
# '0b1000'
```



### int没有想象的那么简单

```python
# base指定进制
int(x, base = 10)
int('100', 2)
# 4
```



### 矩阵转置

```python
 a = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
 list(zip(*a))
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```



### 时间转换

[开挂！](https://www.runoob.com/python/att-time-strftime.html)



### 返回随机值

```python
import random
# 从列表中随机选一个出来
random.choice([1, 2, 3, 4])
# 4
# 从range(3)中随机选一个
random.randrange(3)
# 2
```



### 字符串表达式

```python
# eval()
x = 8
eval(f'3 * {x}')
# 24
```



### get()

```python
# get(value1, value2)
# 第一个参数是键值对的key，第二个参数是如果key不存在返回的默认值
```

