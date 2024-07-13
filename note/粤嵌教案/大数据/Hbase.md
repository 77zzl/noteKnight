# Hbase

shell操作

```sql
# 查看表
list

# 创建表：表名，列族名
create 'student','info'

# 插入数据/更改数据：表名，rowkey，列族名 + 列名，列值
put 'student','1001','info:sex','male'

# 查看全表数据
scan 'student'

# 查询对应行的数据，主意一定要大写
scan 'student',{STARTROW => '1001',STOPROW => '1003'}

# 获取数据，最大范围指定rowkey，即可获取行数据或列数据
get 'student','1001','info:name'

# 计数，仅计数表内多少rowkey
count 'student'

# 删除某rowkey数据
deleteall 'student','1001'

# 删除某一列数据
delete 'student','1002','info:sex'

# 清空表，注意一定要先停用再清空
disable 'student'
truncate 'student'

# 删除表
disable 'student'
drop 'student'

# 查看命名空间
list_namespace

# 创建命名空间
create_namespace 'gec'

# 在命名空间中扔表
create 'gec:emp','info','detail'

# 删除命名空间
disable 'gec:emp'
drop_namespace 'gec'
```

