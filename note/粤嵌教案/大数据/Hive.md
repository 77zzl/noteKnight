### Hive

***

Hive只改变了hdfs呈现数据的方式



###### 服务端与客户端开启

```
# 服务端开启
hiveserver2

# 客户端开启
beeline
!connect jdbc:hive2://hadoop120:10000
```

客户端开启前需要检车端口是否已打开

```
netstat -nptl | grep 10000
```



#### 基本操作

***

###### 操作hdfs文件

```
# 查询文件目录
dfs -ls /;

# 创建及删除文件夹
dfs -mkdir <path>;
dfs -rmdir <path>;

# 复制与移动
dfs -cp <src> <dst>;
dfs -mv <src> <dst>;
```



###### 插入结构体数据

```mysql
create table test(
    name string,
    friends array<string>,
    children map<string, int>,
    address struct<street:string, city:string>
)
# 各字段按','分开
row format delimited fields terminated by ','
# 字段内元素按'_'分开
collection items terminated by '_'
# 字典按':'分隔
map keys terminated by ':'
# 记录间按'\n'换行符分隔
lines terminated by '\n';
```

上例数据如下

```
songsong,bingbing_lili,xiao song:18_xiaoxiao song:19,hui long guan_beijing
yangyang,caicai_susu,xiao yang:18_xiaoxiao yang:19,chao yang_beijing
```



###### 文件导入表中

```mysql
# 从Linux中导入
load data local inpath '/export/servers/datas/test.txt' into table test;

# 从hdfs中导入
load data inpath '/user/test.txt'
```



###### 使用数据库

```mysql
use <database name>;
```



###### 删除数据库

```mysql
# if exists 避免不存在该数据库而报错
# cascade 避免数据库不为空报错
drop database if exists <database name> cascade;
```



###### 内部表与外部表

```mysql
# 内部表可能导致数据丢失
create table test(
	blah blah ...
)
location '/user/hive/warehouse/student2';

# 外部表可以解决这个问题
create external table test_external(
	blah blah ...
)
location '/user/hive/warehouse/student2';

# 将内部表转换为外部表
alter table <table name> set tblproperties('EXTERNAL'='TRUE');
```



#### 数据导入

***

###### 文件导入

```mysql
load data [local] inpath '/export/servers/datas/student.txt' [overwrite] into table student [partition (partcol1=val1,…)];
```

local：是否从本地导入

overwrite：是否追加



###### 基本模式插入

```mysql
insert overwrite table <table name> patrition(month='202110') select id, name from student where month='202110';
```



###### 多表多分区插入模式

```mysql
from student 
insert overwrite table student partition(month='202010') select id, name where month='202110'
insert overwrite table student partition(month='202011') select id, name where month='202111';
```



###### 查询语句时创建表并加载数据

```mysql
create table if not exists <table name> as select id, name from <table name2>;
```



###### 数据导入

```mysql
import table <table name> partition(month='202110') from '<path>';
```



#### 数据导出

***

```mysql
insert [overwrite] [local] directory '<path>' select * from <table name>;
```



#### 分区表

***

分区的概念即分目录，在存储时按参数传入不同的分区中



###### 创建分区表

```mysql
create table <table name>(
	deptno int,
    dname string,
    loc string
)
partitioned by (month string)	# 分区字段
row format delimited fields terminated by '\t';
```



###### 数据导入

```mysql
# 在数据导入时表明分区位置
load data local inpath '/export/servers/datas/dept.txt' into table <database name>.<table name> partition(month='202110');
```



###### 按分区查询

```mysql
select * from <table name> where month='202110';
```



###### 增加分区

```mysql
alter table <table name> add partition(month='202111');
```



###### 查看分区

```mysql
show partitions <table name>;
```



###### 上传文件并创建分区

方法一

```mysql
# 创建文件夹
dfs -mkdir -p /user/hive/warehouse/db_hive.db/dept_partition/month=202110/day=11;

# 上传文件
dfs -put /export/servers/datas/dept.txt /user/hive/warehouse/db_hive.db/dept_partition2/month=202110/day=11;

# 添加分区
alter table dept_partition2 add partition(month='202110', day='11');
```

方法二

```mysql
# 创建文件夹
dfs -mkdir -p /user/hive/warehouse/db_hive.db/dept_partition/month=202110/day=11;

# 上传数据
load data local inpath '/export/servers/datas/dept.txt' into table dept_partition(month='202110', day='10');
```





#### bucket

***

方便进行抽样调查，可以对数据集有一个整体把握

分区表比分桶表更常见！



###### 创建分桶表

```mysql
# clustered by(<name>) into <num> buckets表明分桶字段，以及桶数
create table buck(id int, name string)
clustered by(id)
into 4 buckets
row format delimited fields terminatied by '\t';
```



###### 属性设置

```mysql
# 强制分桶
set hive.enforce.bucketing=true;
# 自动划分reducer个数
set mapreduce.job.reduces=-1;
```



###### 分桶规则

数据模哈希值，例如本例中16个数据模4就是4 * 4，分成四个桶每个桶四个数据，并按照指定的id值进行分桶



###### 抽样查询

抽样语句

```mysql
# 将哈希值除以y，得到每个桶要抽取的个数，再从第x个桶开始抽取，可以指定字段抽取
tablesample(bucket <x> out of <y> [on <name>])
```

实例

```mysql
# 抽取4 / 2，即每个桶抽两个数据，总共2 * 4个数据，并从第一个桶开始抽取
select * from buck tablesample(bucket 1 out of 2 on id);
# 抽取4 / 8，即每个桶抽二分之一，这里可以理解为两个桶抽取一个数据，总共1/2 * 4个数据，并从第三个数据开始
select * from buck tablesample(bucket 3 out of 8 on id);
# 抽取4 / 4，即每个桶抽一个数据，总共4 * 1个数据
select * from buck tablesample(bucket 1 out of 4);
```



#### 常用函数

***

###### nvl

```mysql
# 空字段赋值，将comm为空的数据替换为-1
select comm, nvl(comm, -1) from emp;
# 用其他字段补全
select comm, nvl(comm, mgr) from emp;
```



###### case when

```mysql
# 基本语法
case <字段> when <value> then <DoSomething> else <DoOtherThing> end
# 例如计算某字段个数
select dept_id,
sum(case sex when '男' then 1 else 0 end) male_count,
sum(case sex when '女' then 1 else 0 end) female_count
from emp_sex group by dept_id;
```



###### concat

```mysql
# 张三 is male
concat(name, ' is ', sex)
```



###### collect_set

```mysql
# collect_set
select
    t1.base,
    collect_set(t1.name)
from
    (select
        name,
        concat(constellation, ",", blood_type) base
    from
        person_info) t1
group by
    t1.base;
```

分析这条语句，group by 保证按照concat的值将base相同的分为一组，collect_set将名字拼接在一起，可以把collect_set理解为另一种形式的concat，区别在于concat是连接不同字段，而collect_set 为连接同种字段



###### concat_ws

```mysql
# 将group在一起的name按照'|'分割开来
concat_ws('|', collect_set(name))
```



###### explode

```mysql
# 将一行分成多列
select
    movie,
    category_name
from 
    movie_info 
lateral view
    explode(category) table_tmp as category_name;
```

lateral view 表示侧视图，单独的explode(category) 为一个表，别名为table_tmp 并将此表作为category_name

```mysql
from
(select
    m.movie,
    table_tmp.category_name
from 
    movie_info m
lateral view
    explode(category) table_tmp as category_name) t
select
    category_name,
    concat_ws(",", collect_set(movie)) movies
group by
    category_name;
```

分析一下这个语句，Hive允许from ... select 的形式，便于对表结构进行处理再选择所需字段，再将已分离的数组作为新的表结构，此时再用concat_ws和collect_set将movie按所需呈现



###### substring

```mysql
# 分隔字符串，将str从第a个字符开始往后b个
substring(str, a, b)	
```



#### 窗口函数

***

为聚合函数专门设置一个窗口，用以简易的分析并显示数据，或者理解为为聚合函数进行一次升级

```mysql
# 在开始前首先理解一个概念
# 窗口函数是sql语句最后执行的函数，因此可以将sql的结果想象成传进窗口函数的数据
# 所谓的窗口函数就是over()，它建立在聚合函数的基础上
```



直接实战

###### 聚合函数 + over

```mysql
# 单独的count(*)仅有一行，增加over() 之后将会显示人名并在每个人名后写入count(*)数据
select name, count(*) over()
from business;

# 可将上述方法升级为
select distinct name, count(*) over()
from business;
# 或
select name, count(*) over()
from business
group by name;
```

```
name    count_window_0
mart    5
mart    5
mart    5
mart    5
jack    5
```



###### 窗口函数的小跟班

```mysql
# 除了按需呈现所需的聚合函数外，窗口函数也可以按需进行排序分组

# partition by

# 这里注意，聚合函数按照分区字段分区后，sum()将单独计算每个分区内的值，因此在查询结果中，聚合函数显示的值并不一致

select name, orderdate, cost, sum(cost) over(partition by month(orderdate))
from business;


# order by

# 该处理相当于在上述处理的基础上，对每一个分区进行了排序并累加

select name, orderdate, cost, sum(cost) over(partition by month(orderdate) order by orderdate) from business;
```

```
name    orderdate   cost    sum_window_0

jack    2015-01-01  10  	10
tony    2015-01-02  15  	25 //10+15
tony    2015-01-04  29  	54 //10+15+29
jack    2015-01-05  46  	100 //10+15+29+46
tony    2015-01-07  50  	150
jack    2015-01-08  55  	205
jack    2015-02-03  23  	23
jack    2015-04-06  42  	42
mart    2015-04-08  62  	104
mart    2015-04-09  68  	172
mart    2015-04-11  75  	247
mart    2015-04-13  94  	341
neil    2015-05-10  12  	12
neil    2015-06-12  80  	80
```



###### WINDOW

```mysql
# 更精细的分组处理！

# 当一个语句中存在多个窗口函数时，每个窗口函数都应该应用自己的规则互不影响

# window子句 rows between [] and []

# 仅有order by 时默认做组内排序并累加，当有window子句时才对每一段做独立处理

# preceding 往前
# following 往后
# current row 当前行
# unbounded 起点|终点

# 注意一点，因为window子句是最晚执行的，因此unbounded所指向的起点并非数据起点，而是组内起点

select name,orderdate,cost,
sum(cost) over() as sample1,	# 所有行的sum()值相加
sum(cost) over(partition by name) as sample2,	# 按name分组，组内数据相加
sum(cost) over(partition by name order by orderdate) as sample3,	# 按name分组，组内数据累加
sum(cost) over(partition by name order by orderdate rows between UNBOUNDED PRECEDING and current row )  as sample4 ,	# 和sample3一样,由起点到当前行的聚合
sum(cost) over(partition by name order by orderdate rows between 1 PRECEDING   and current row) as sample5,		# 当前行和前面一行做聚合
sum(cost) over(partition by name order by orderdate rows between 1 PRECEDING   AND 1 FOLLOWING  ) as sample6,	# 当前行和前边一行及后面一行
sum(cost) over(partition by name order by orderdate rows between current row and UNBOUNDED FOLLOWING ) as sample7 	# 当前行及后面所有行，相当于反向累加
from business;
```

```
name    orderdate   cost    sample1 sample2 sample3 sample4 sample5 sample6 sample7

jack    2015-01-01  10  	661 	176 	10  	10  	10  	56  	176
jack    2015-01-05  46  	661 	176 	56  	56  	56  	111 	166
jack    2015-01-08  55  	661 	176 	111 	111 	101 	124 	120
jack    2015-02-03  23  	661 	176 	134 	134 	78  	120 	65
jack    2015-04-06  42  	661 	176 	176 	176 	65  	65  	42
mart    2015-04-08  62  	661 	299 	62  	62  	62  	130 	299
mart    2015-04-09  68  	661 	299 	130 	130 	130 	205 	237
mart    2015-04-11  75  	661 	299 	205 	205 	143 	237 	169
mart    2015-04-13  94  	661 	299 	299 	299 	169 	169 	94
neil    2015-05-10  12  	661 	92  	12  	12  	12  	92  	92
neil    2015-06-12  80  	661 	92  	92  	92  	92  	92  	80
tony    2015-01-02  15  	661 	94  	15  	15  	15  	44  	94
tony    2015-01-04  29  	661 	94  	44  	44  	44  	94  	79
tony    2015-01-07  50  	661 	94  	94  	94  	79  	79  	50
```



###### NTILE

```mysql
# 用于将分组数据按照顺序切分成n片，返回当前切片值
# 如果切片不均匀，默认增加前几个切片的分布
# 切片可以用来实现按需呈现所需数据的某个部分
select name,orderdate,cost,
       ntile(3) over() as sample1 ,						# 全局数据切片
       ntile(3) over(partition by name) as sample2, 	# 按照name进行分组,在分组内将数据切成3份
       ntile(3) over(order by cost) as sample3,			# 全局按照cost升序排列,数据切成3份
       ntile(3) over(partition by name order by cost ) as sample4 		# 按照name分组，在分组内按照cost升序排列,数据切成3份
from t_window
```

```
name    orderdate   cost    sample1 sample2 sample3 sample4

jack    2015-01-01  10  	3   	1   	1   	1
jack    2015-02-03  23  	3   	1   	1   	1
jack    2015-04-06  42  	2   	2   	2   	2
jack    2015-01-05  46  	2   	2   	2   	2
jack    2015-01-08  55  	2   	3   	2   	3
mart    2015-04-08  62  	2   	1   	2   	1
mart    2015-04-09  68  	1   	2   	3   	1
mart    2015-04-11  75  	1   	3   	3   	2
mart    2015-04-13  94  	1   	1   	3   	3
neil    2015-05-10  12  	1   	2   	1   	1
neil    2015-06-12  80  	1   	1   	3   	2
tony    2015-01-02  15  	3   	2   	1   	1
tony    2015-01-04  29  	3   	3   	1   	2
tony    2015-01-07  50  	2   	1   	2   	3
```



###### 排序函数

```mysql
# 下述所有函数都将对order by的字段进行排序，区别均在于对排名相同的数据进行不同的处理

# ROW_NUMBER
# row_number()对相同排名的记录将按照显示的顺序进行排名，排名名次不会重复

# RANK
# rank()对相同排名的记录将给以同样的名次，但会给相同名次的记录留下空位，rank无法保证名次连续

# DENSE_RANK
# dence_rank()对于相同排名的记录将给以同样的名次，但不会给同名次的记录留下空位，dense_rank将保证名次连续
SELECT 
cookieid, createtime, pv,
RANK() OVER(PARTITION BY cookieid ORDER BY pv desc) AS rn1,
DENSE_RANK() OVER(PARTITION BY cookieid ORDER BY pv desc) AS rn2,
ROW_NUMBER() OVER(PARTITION BY cookieid ORDER BY pv DESC) AS rn3 
FROM business;
```

```
cookieid day           pv       rn1     rn2     rn3 
 
cookie1 2015-04-12      7       1       1       1
cookie1 2015-04-11      5       2       2       2
cookie1 2015-04-15      4       3       3       3
cookie1 2015-04-16      4       3       3       4
cookie1 2015-04-13      3       5       4       5
cookie1 2015-04-14      2       6       5       6
cookie1 2015-04-10      1       7       6       7
```

```
TIPS：

　　使用rank over()的时候，空值是最大的，如果排序字段为null, 可能造成null字段排在最前面，影响排序结果。

　　可以这样： rank over(partition by course order by score desc nulls last)
```

```
在使用排名函数的时候需要注意以下三点：

　　1、排名函数必须有 OVER 子句。

　　2、排名函数必须有包含 ORDER BY 的 OVER 子句。

　　3、分组内从1开始排序。
```



###### LAG & LEAD

```mysql
# lag
# 向上取第n次的记录，可传入三个参数，分别为所查询的字段、向上第n条记录、第一个数据若为null的默认值
# lead与lag相反，向下取第n条记录
select name,orderdate,cost,
lag(orderdate,1,'1900-01-01') over(partition by name order by orderdate ) as time1,
lag(orderdate,2) over (partition by name order by orderdate) as time2
from business;
```

```
name    orderdate   cost    time1   	time2

jack    2015-01-01  10  	1900-01-01  NULL
jack    2015-01-05  46  	2015-01-01  NULL
jack    2015-01-08  55  	2015-01-05  2015-01-01
jack    2015-02-03  23  	2015-01-08  2015-01-05
jack    2015-04-06  42  	2015-02-03  2015-01-08
mart    2015-04-08  62  	1900-01-01  NULL
mart    2015-04-09  68  	2015-04-08  NULL
mart    2015-04-11  75  	2015-04-09  2015-04-08
mart    2015-04-13  94  	2015-04-11  2015-04-09
neil    2015-05-10  12  	1900-01-01  NULL
neil    2015-06-12  80  	2015-05-10  NULL
tony    2015-01-02  15  	1900-01-01  NULL
tony    2015-01-04  29  	2015-01-02  NULL
tony    2015-01-07  50  	2015-01-04  2015-01-02
```



###### first & last

```mysql
# first_value
# 取分组内排序后，截止到当前行，第一个值

# last_value
# 取分组内排序后，截止到当前行，最后一个值(相当于自己??)
select name, orderdate, cost,
first_value(orderdate) over(partition by name order by orderdate) as time1,
last_value(orderdate) over(partition by name order by orderdate) as time2
from business;
```

```
name    orderdate   cost    time1   	time2
jack    2015-01-01  10  	2015-01-01  2015-01-01
jack    2015-01-05  46  	2015-01-01  2015-01-05
jack    2015-01-08  55  	2015-01-01  2015-01-08
jack    2015-02-03  23  	2015-01-01  2015-02-03
jack    2015-04-06  42  	2015-01-01  2015-04-06
mart    2015-04-08  62  	2015-04-08  2015-04-08
mart    2015-04-09  68  	2015-04-08  2015-04-09
mart    2015-04-11  75  	2015-04-08  2015-04-11
mart    2015-04-13  94  	2015-04-08  2015-04-13
neil    2015-05-10  12  	2015-05-10  2015-05-10
neil    2015-06-12  80  	2015-05-10  2015-06-12
tony    2015-01-02  15  	2015-01-02  2015-01-02
tony    2015-01-04  29  	2015-01-02  2015-01-04
tony    2015-01-07  50  	2015-01-02  2015-01-07
```



#### 自定义函数

***

###### 类别

1. UDF（User-Defined-Function）

   一进一出

2. UDAF（User-Defined Aggregation Function）

   聚集函数，多进一出

3. UDTF（User-Defined Table-Generating Functions）

   一进多出

[官网地址](https://cwiki.apache.org/confluence/display/Hive/HivePlugins)



###### 编程步骤

1. 导入apache.hive的依赖

2. 继承apache的UDF类，类名即为hive中的函数名

3. 实现evaluate函数，该函数支持重载

4. 在hive中创建函数并与jar包关联

   

   ```mysql
   # 添加jar包
   add jar /export/servers/hive/lib/udf.jar;
   
   # 创建函数并与Jar包关联
   create temporary function mylower as "com.zhang.hive.Lower";
   
   # 使用
   select *,mylower(name) lowername from student;
   ```

   

