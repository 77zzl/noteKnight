### [thrift](https://thrift.apache.org/tutorial/)

***

rpc框架：调用其他服务器的函数



### Start

***

1.create a thrift file

2.This file is an [interface definition](https://thrift.apache.org/docs/idl) made up of [thrift types](https://thrift.apache.org/docs/types) and Services

3.Thrift compiler is used to generate your Thrift File into source code which is used by the different client libraries and the server you write

```
thrift --gen <language> <Thrift filename>
```

简单来说！当使用同一份.thrift脚本生成的客户端与服务端文件时，将自动进行连接



### Tutorial

***

#### type

```
/**
    * The first thing to know about are types. The available types in Thrift are:
    *
    *  bool        Boolean, one byte
    *  i8 (byte)   Signed 8-bit integer
    *  i16         Signed 16-bit integer
    *  i32         Signed 32-bit integer
    *  i64         Signed 64-bit integer
    *  double      64-bit floating point value
    *  string      String
    *  binary      Blob (byte array)
    *  map<t1,t2>  Map from one type to another
    *  list<t1>    Ordered list of one type
    *  set<t1>     Set of unique elements of one type
    *
    * Did you also notice that Thrift supports C style comments?
    */
```



#### namespace

```
namespace <language> tutorial
```

Just in case you were wondering... You can click [here](https://www.runoob.com/cplusplus/cpp-namespaces.html) to remember what a namespace is



#### typedef

Thrift lets you do typedefs to get pretty names for your types. Standard C style here.

```
typedef i32 MyInteger
```



#### constants

Thrift also lets you define constants for use across languages. Complex types and structs are specified using JSON notation.

```
const i32 INT32CONSTANT = 9853

const map<string,string> MAPCONSTANT = {'hello':'world', 'goodnight':'moon'}
```



#### Enum and Struct

Note that the struct has integer identifier, while the enum does not

```
enum Operation {
     ADD = 1,
     SUBTRACT = 2,
     MULTIPLY = 3,
     DIVIDE = 4
}
```

I know you are not smart, so if you forget what enum is, please click [here](https://www.runoob.com/cprogramming/c-enum.html)

```
struct Work {
   1: i32 num1 = 0,
   2: i32 num2,
   3: Operation op,
   4: optional string comment,
}
```



### let's do it

***

#### 需求

1、实现客户端，匹配函数（match-cli）

2、实现服务端，匹配服务（match-ser）、保存数据（save-cli）



#### mkdir

match_system：服务端

game：客户端

thrift：接口



#### 服务端

0、下面这堆东西叫接口

1、定义命名空间

```
namespace cpp match_service
```

2、数据用结构体定义

```
struct User {
	1: i32 id,
	2: string name,
	3: i32 score
}
```

3、函数要放在 **service** 里面

```
service Match {
	i32 add_user(1:User user, 2:string info),
	
	i32 remove_user(1:User user, 2:string info)
}
```

4、进入要创建节点的文件下，创建一个 **src** 表示源文件，并进入

5、执行

```
thrift -r --gen cpp <.thrift文件的地址>
```

6、发现在当前目录下出现 **gen-cpp** 文件，将其改名为 **match_server**

7、将 **match_server** 文件下的 **Match_server.skeleton.cpp** 移动到 **src** 下并改名为 **main.cpp**

8、修改 **main.cpp** 保证能编译通过（添加return 0、将引用文件目录地址修改正确Match.h、添加输出语句）

9、编译、链接

```
g++ -c main.cpp match_server/*.cpp

g++ *.o -o main -lthrift -thread
```

10、运行可执行文件

```
./main
```

:star:如果是 **python** 语言生成的 **thrift** 则生成名为 **Match-remote** 的服务器文件

注意上传时，移除 **.o** 文件和可执行文件 **main**

```
git restore --stage *.o
git restore --stage main
```



#### 客户端

1、在 **game** 目录下创建 **src** 目录并创建 **python** 代码

```
thrift -r --gen py ../../thrift/match.thrift
```

2、删除服务端文件 **Match-remote** 

3、在 **src** 目录下写客户端代码 **client.py** ，并自己拷下[客户端](https://thrift.apache.org/tutorial/py.html)

4、改写

5、编译

```
python3 client.py
```

