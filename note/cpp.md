# C++

## 面向对象

为了满足面对相对象的需要，c++具有许多c没有的特性。接下来将一一列举，有许多特性在刚接触的时候可能一头雾水，但是随着学习的推进也将慢慢明白它在工程中的作用。尽管c++存在一些问题，但永远不要怀疑这门语言的强大！

本文仅记录c++异于c的特性！

### 引用

c++对许多运算符进行了重载操作，常见的有`<<`和`>>`，将原本适用于二进制计算的运算符用于输入输出。引用也是其中一个例子，在c中`&`符号的意思是取地址，而c++会根据上下文语义赋予它新的含义——引用。

引用可以理解成对指针解除引用的替身，比如我们希望函数修改一个变量的值：

```c++
// c
void swap(int *a, int *b)
{
  	int tmp = *a;
  	*a = *b;
  	*b = tmp;
}

// c++
void swap(int &a, int &b)
{
  	int tmp = a;
  	a = b;
  	b = a;
}
```

是的可以少写一个小星星。

假如我们想要创建一个引用`int &p = a`，此时我们相当于为a创建了一个别名p，当我们修改p时a也会随之修改。这意味着a和p指向同一块地址，也就是说引用其实不过是一串不需要解引用就能取值的地址罢了！

使用引用的场景：

- 对于类和结构这些庞大的数据结构时使用引用可以为运行加速
- 当我们在范围循环或者函数传参时希望修改值时

特别指出一点：有时候会见到`const & `，那么开发人员的意图是传递一个较大的数据结构(类或者结构)，但不希望函数修改其值，在编码时尽可能的写`const`是一个好习惯！

<br>

### 函数重载

- 函数特征标
  - 取决于函数的参数列表
  - 不同的特征标
    - 参数列表长度不同
    - 参数列表内元素的类型任一存在不同

- 函数重载指的就是，c++允许函数特征标不同但是函数名相同

c++内表示数值的数据类型众多，当我们在实现一个函数时，比如说两数之和，我们需要处理的不仅仅是int类型的求和还有float还有long，但我们不可能专门为它们编写一个独立的函数，这样会使得我们的函数冗余且使用起来也不方便。因此，当我们编写了一个函数后，我们完全可以使用相同的函数名，仅仅修改参数列表和内部元素的类型创建新的符合要求的函数。

```c++
void swap(int &a, int &b)
{
  	int tmp = a;
  	a = b;
  	b = a;
}

void swap(float &a, float &b)
{
  	float tmp = a;
  	a = b;
  	b = tmp;
}
```

可是这样并不能满足我们的需求，因为这并未减少我们的代码量，仅仅是帮助程序员少起一个函数名而已。

- 函数模版
  - 我们制定一个模版，让计算机自己根据判断数据类型来帮我们编写函数

```c++
// typename向后兼容，class向前兼容
template <typename T>
void swap(T &a, T &b)
{
  	T tmp = a;
  	a = b;
  	b = tmp;
}
```

该程序的作用是：当我们调用函数swap时，程序自动判断参数类型，如果两个参数一样则只用该swap函数，并且tmp的类型也将与参数的类型保持一致，使用函数重载+函数模版我们可以大大减少编程工作。这也使得我们生成多个函数定义更加简单、更可靠。

**特别注意：模版并非函数定义，仅仅是提供了一种生成函数定义的方案**，只用当我们使用该函数时，编译器才会帮我们完成定义。

- 显式具体化
  - 有时候我们使用模版定义了这个函数，但是实际上这个函数并不能处理所有的类型(结构)，如果盲目让程序为其定义函数可能会导致意料之外的错误，因此针对这种特殊情况c++提出了显式具体化的概念
  - 为特定类型提供具体化的模版定义

```c++
struct job
{
    char name[40];
    double salary;
    int floor;
}

// template <> void swap(job &a, job &b) 同样允许
template <> void <job>swap(job &a, job &b) {};

template <typename T>
void swap(T &a, T &b) {};
```

这时候，如果我们在调用swap函数时使用的参数是job结构，那么则会调用显式具体化的函数。

<br>

<br>

## 标准模版库

先来介绍所有容器都有的`api`

- size()
  - O(1)时间返回容器大小
- empty()
  - O(1)时间返回是否为空

### vector

变长数组，根据倍增思想拓容

注意仅支持从尾端插入元素与取出元素

自带比较函数

```c++
#include <vector>
#include <iostream>

using namespace std;

int main()
{
    vector<int> a;

    // 插入数据
    for (int i = 0; i < 10; i ++) a.push_back(i);

    /* 查询
    查询第一个元素
    查询最后一个元素
    索引查询
    */
    cout << a.front() << endl;
    cout << a.back() << endl;
    cout << a[0] << endl;

    // 弹出最后一个数据
    a.pop_back();

    // 迭代器
    for (auto i = a.begin(); i != a.end(); i ++) cout << *i << ' ';
    cout << endl;
}
```

<br>

### pair

二元组，`pair`无需导入任何头文件即可使用

自带比较函数，已实现结构体

```c++
#include <cstring>
#include <iostream>

using namespace std;

int main()
{
    // 初始化
    pair<int, string> p;

    // 赋值
    p = {77, "zzl"};
    p = make_pair(77, "zzl");

    // 取值
    cout << p.first << " " << p.second << endl;

    // 假如我们比较贪心，也可以存三元组
    pair<int, pair<int, int>> p1;

    return 0;
}
```



<br>

### string

字符串，有点python那味了

```c++
#include <cstring>
#include <iostream>

using namespace std;

int main()
{
    // 初始化
    string a = "zzl";

    // 新增
    a = "77" + a;

    // 切片
    cout << a.substr(2, 3) << endl;
    cout << a.substr(2) << endl;

    /* 
    string转char
    string只有转变为char才能printf输出
    众所周知printf的输出效率快的不是一点
    c_str()返回字符串开头
    */
    printf("%s\n", a.c_str());
  	const char* str = a.c_str();
  
  	// char转string
  	char* c = "zzl";
  	string s = c;

    return 0;
}
```

<br>

### queue

队列，先进先出

```c++
#include <iostream>
#include <queue>

using namespace std;

int main()
{
    // 初始化
    queue<int> q;

    // 队尾插入
    for(int i = 0; i < 4; i ++) q.push(i);

    // 队首弹出
    q.pop();

    /*
    查询
    返回队首返回队尾
    */
    cout << "队首:" << q.front() << " 队尾:" << q.back() << endl;

    return 0;
}
```

<br>

### priority_queue

大根堆，弹出最大值

```c++
#include <iostream>
#include <queue>

using namespace std;

int main()
{
    priority_queue<int> heap;

    // 插入
    for (int i = 5; i > 0; i --) heap.push(i);

    // 弹出堆顶元素
    heap.pop();

    // 获取堆顶元素
    cout << heap.top() << endl;

    return 0;
    /*
    小根堆
    方法一：
      前提是所有数据都是正整数
      将所有插入的数取负号即可
    方法二：
      定义的时候声明为小根堆
      prioriy_queue<int, vector<int>, greater<int>> heap;
    */
}
```

<br>

### stack

栈，先进后出

```c++
#include <iostream>
#include <stack>

using namespace std;

int main()
{
    // 初始化
    stack<int> s;

    // 插入
    for (int i = 0; i < 4; i ++) s.push(i);

    // 弹出
    s.pop();

    // 查询栈顶
    cout << s.top() << endl;

    return 0;
}
```

<br>

### duque

双端队列，加强版`vector`，速度特别慢

```c++
#include <iostream>
#include <deque>

using namespace std;

int main()
{
    /*
    vector有的它都有
    它还有vector没有的
    前端和后端都可以插入弹出
    */
    deque<int> d;

    // 插入 
    d.push_front(5);
    d.push_back(5);

    // 弹出
    d.pop_front();
    d.pop_back();

    return 0;
}
```

<br>

### set

集合，不允许有重复元素

操作的时间复杂度均为O(logn)

- multiset
  - 与set类似但是支持重复元素

```c++
#include <set>
#include <iostream>

using namespace std;

int main()
{
    set<int> s;

    // 插入
    for (int i = 0; i < 4; i ++) s.insert(i);

    /* 
    查找
    不存在返回迭代器end()
    注意迭代器不同c中地址不可被输出
    */
    cout << "find 2: " << *s.find(2) << endl;
    cout << "find 8: " << *s.find(8) << endl;

    // 计数，只有0和1
    cout << "统计 1: " << s.count(1) << endl;

    /*
    删除
    参数为一个数则删除该数
    参数为迭代器则删除迭代器
    */
    s.erase(2);

    /*
    单调栈？
    返回的值是一个迭代器
    不存在返回end()
    lower_bound(x)返回大于等于x最小的数
    upper_bound(x)返回大于x最小的数
    */
    cout << "lower_bound 1: " << *s.lower_bound(1) << endl;
    cout << "upper_bound 1: " << *s.upper_bound(1) << endl;

    return 0;
}
```

<br>

### map

字典，数据具有映射关系，key-value键值对，每个key只能有一个

和`python`的字典使用方式很像

注意几乎所有操作的时间都是O(logn)

它虽然具有映射关系但不是二元组而是结构体，无法使用`.first`只能使用`->first`取出数据

操作的时间复杂度均为O(logn)

- multimap
  - 与map类似但是支持重复key

```c++
#include <iostream>
#include <cstdio>
#include <map>

using namespace std;

int main()
{
    // 初始化
    map<string, int> a;
    a["zzl"] = 1;
    map<int,string> b = {{1, "dh"}, {2, "xm"}, {3, "eg"}};
    map<int,string> c = {pair<int,string> (1, "java"), pair<int, string> (2, "c++")};
  
    // 插入
    b.insert(pair<int, string> (1, "ggboy"));
    c.insert({2, "yygril"});

    /* 
    删除
    按照key来删除
    按照迭代器删除，将返回下一个元素的迭代器
    */
    b.erase(2);

    // 遍历
    for (auto it = b.begin(); it != b.end(); it ++)
        cout << it->first << " " << it->second << endl;

  	// 计数，非0即1
  	cout << "count 3: " << b.count(3) << endl;
  
  	/*
  	查找
  	key找value直接使用数组索引的方式
  	但是如果不存在key则会发生奇怪的错误，因此需要使用find来判断
  	find如果找不到会返回最后一位
  	当然了判断是否存在也可以用count来判断
  	*/
  	if (b.find(3) == b.end())
      cout << "404!" << endl;
  	else
      cout << b.find(3) -> second << endl;
  
    return 0;
}
```

<br>

### unordered

以下容器与原容器功能基本一样，操作速度可达O(1)

**但是不支持lower_bound/upper_bound**

- unordered_set

- unordered_map

- unorder_multiset

- unorder_multimap

<br>

### bitset

占用空间巨的比特集，一个元素一比特

支持所有位运算操作

注意bitset不支持所有非位运算的运算符

因为计算上的缺陷一般仅用于数据压缩

- size()
  - 有多少位
- count()
  - 有多少个1
- all()
  - 判断是否全为1
- any()
  - 判断是否至少一个1
- none()
  - 判断是否全为零

```c++
#include <iostream>
#include <bitset>

using namespace std;

int main()
{
    const int size = 4;
    bitset<size> a("0101");
  	bitset<size> b(5);
    cout << "size: " << a.size() << endl;
    cout << "count: " << a.count() << endl;
    cout << "any: " << a.any() << "\nall: " << a.all() << endl;
    
    // 所有位取反
    a = ~a;
    cout << a << endl;
    
    /* 
    所有位1取0
    两种方法实现
    注意^只可以和bitset类型使用
    */
    a ^= a;
    a &= 0;
    cout << a << endl;
    
    // 所有位取1
    a ^= ~a;
    cout << a << endl;
    
    //返回一个unsigned long值
    cout << a.to_ulong() << endl;  
    //返回一个unsigned long long 值
    cout << a.to_ullong() << endl;  
    //返回一个string
    cout << a.to_string() << endl;  
  
    return 0;
}
```

