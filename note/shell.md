# Shell



#### 脚本语言

***

```shell
# 开头要写上
#! /bin/bash

# 赋予可执行权限
chmod +x test.sh
```



#### 注释

***

```shell
# 单行注释

:<<anywords
第一行注释
第二行注释
第三行注释
anywords
```



#### 变量

***

###### 字符串

```shell
# 字符串的定义比较随意
# 单引号中的内容将原样输出，双引号则会取值
name1 = 'zzl'
name2 = "zzl"
name3 = zzl

# 获取字符串长度
echo ${#name1}
:<<out
3
out

# 提取子串
echo ${name1:0:1}
:<<out
zz
out

# 字符串长度
length $name

# 某字符在字符串中的坐标
# 返回name中a的坐标1，如果没有则返回0
index "$name" abc

# 子串函数
# 返回name中第2个字符开始的后三位
substr "$name" 2 3
```



###### 变量

```shell
# 变量的使用
echo $name1
:<<out
zzl
out

echo ${name1}IsAGuy
:<<out
zzlIsAGut
out

# 只读变量
readonly name1

# 删除变量
unset name1

# 自定义变量 -> 环境变量
export name1
# 或
declare -x name1

# 环境变量 -> 自定义变量
declare +x name1
```



#### 文件变量参数

***

```shell
# 脚本运行时可向脚本传递参数
./test.sh 1 2 3

# 文件名
# test.sh
echo $0

# 第一个参数
# 1
echo $1

# 第二个参数
# 2
echo $2

# 参数的个数
# 3
echo $#

# 列举所有传输的参数
# 1 2 3
echo $*

# 脚本当前的进程ID
echo $$

# 返回上一条命令的退出状态
echo $?

# 返回命令的输出(stdout)
$(command)
# 或
`command`
```



#### 数组

***

```shell
# shell仅支持一维数组
# 数组的定义
array=(1 abc 'acd' "zzl")

# 或者
array[0]=1
array[1]='zzl'

# 读取数组元素
${array[index]}

# 读取数组全部内容
$array[*]
```



#### expr

***

###### 观察这条语句

```shell
echo `expr length "$str"`
```

其中`expr`命令表示执行表达式，``则返回expr命令的输出，即执行结果



###### 需要转义的字符

乘号 `\*`

双括号 `\( 2 + 3 \)`

大于小于等于 `\>` 和 `\<` 和 `\=`

逻辑关系表达式 `\|` 和 `\&`  



#### std

***

#### stdin

```shell
# read用来读取标准输入
# -p 后接提示信息
# -t 后接等待秒数，超过时间跳过此命令
read name -p "我是谁？你有十秒时间思考" -t 10
```



#### stdout

###### echo

```shell
# -e 开启转义
echo -e "hi\n"

# \c 不换行
echo -e "hi\c"

# 显示Linux命令执行结果
echo `date`
```



###### printf

```shell
# shell脚本的格式与C完全一致，区别在于参数用空格隔开
printf format-string [arguments...]

printf "%d * %d = %d\n" 2 3 `expr 2 \* 3`
```



#### test和`'[]'`

***

###### 首先

test 的作用与` '[]'` 一致

注意`'[]'`中的每一项都要用空格隔开

所有常数和变量最好都用`""`括起来

测试语句之间可以用 `&&` 和 `||` 连接



###### test的作用

判断文件类型，以及对变量作比较

需要注意的是返回结果为`exit code`而不是`stdout`因此0为真，非0为假



###### 文件类型判断

| 参数 | 意义         |
| ---- | ------------ |
| -e   | 文件是否存在 |
| -f   | 是否为文件   |
| -d   | 是否为目录   |



###### 文件权限判断

| 参数 | 意义       |
| ---- | ---------- |
| -r   | 是否可读   |
| -w   | 是否可写   |
| -x   | 是否可执行 |
| -s   | 是否为空   |



###### 整数比较

| 参数 | 意义     |
| ---- | -------- |
| -eq  | 等于     |
| -ne  | 不等于   |
| -gt  | 大于     |
| -lt  | 小于     |
| -ge  | 大于等于 |
| -le  | 小于等于 |



###### 字符串比较

| 参数 | 意义         |
| ---- | ------------ |
| -z   | 判断是否为空 |
| -n   | 判断是否空   |
| ==   | 等于         |
| !=   | 不等于       |



###### 多条件判定

| 参数 | 意义 |
| ---- | ---- |
| -a   | &&   |
| -o   | \|\| |
| !    | !    |



###### 例子

```shell
[ -e test.sh ] && echo "exist" || echo "Not exist"
```



#### 判断

***

###### if

```shell
if condition
then
	语句
elif condition
then
	语句
else
	语句
fi
```



**例子**

```shell
a=4

if [ $a -eq 1 ]
then
    echo ${a}等于1
elif [ $a -eq 2 ]
then
    echo ${a}等于2
elif [ $a -eq 3 ]
then
    echo ${a}等于3
else
    echo 其他
fi
```



###### case

```shell
case $变量名称 in
    值1)
        语句1
        语句2
        ...
        ;;  # 类似于C/C++中的break
    值2)
        语句1
        语句2
        ...
        ;;
    *)  # 类似于C/C++中的default
        语句1
        语句2
        ...
        ;;
esac
```



**例子**

```shell
a=4

case $a in
    1)
        echo ${a}等于1
        ;;  
    2)
        echo ${a}等于2
        ;;  
    3)                                                
        echo ${a}等于3
        ;;  
    *)
        echo 其他
        ;;  
esac
```



#### 循环

***

###### for...in

```shell
for var in val1 val2 val3
do 
	语句
done
```

 

**例子**

```shell
for file in `ls`
do
    echo $file
done
```

```shell
for i in $(seq 1 10)
do
    echo $i
done
```

```shell
for i in {a..z}
do
    echo $i
done
```



###### for((...))

```shell
for ((expression; condition; expression))
do
    语句1
done
```



###### while

```shell
while condition
do
    语句1
    语句2
    ...
done
```



###### until

```shell
until condition
do
    语句1
    语句2
    ...
done
```



#### 函数

***

```shell
[function] func_name() {  # function关键字可以省略
    语句
    ...
}
```

 

###### 局部变量

```shell
local 变量名=变量值
```



#### 重定向

***

| 参数            | 说明                             |
| --------------- | -------------------------------- |
| command > file  | 将stdout重定向到file中           |
| command < file  | 将stdin重定向到file中            |
| command >> file | 将stdout以追加方式重定向到file中 |



#### 外部脚本

***

```shell
. filename  # 注意点和文件名之间有一个空格

或

source filename
```

