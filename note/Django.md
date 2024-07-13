# django

梦开始的地方：`pip install django`

Django采用的框架和传统MVC有些不同

### MVT

- model，模型，与数据库交互
- view，视图，逻辑处理
- template，模版，呈现前端内容

### 原理

1. 程序按照用户的需求不同交给不同的**视图**进行处理
2. **视图**按照用户的需求提供不同的服务
3. 如果需要访问数据库，**视图**将使用**模型**来与数据库进行交互
4. 如果需要显示新的页面则返回给用户所需的**模版**

<br>

## Prepare

创建一个django项目：`python3 django createproject project`

创建一个app：`python3 manage.py startapp game`

数据库变更

- `python3 manage.py makemigrations`
- `python3 manage.py migrate`

创建新的管理员：`python3 manage.py createsuperuser`

启动项目：`python3 manage.py runserver 0.0.0.0:8000`

<br>

### 目录

特别注意：如果是某个文件夹下存放python文件，那么该文件夹下必须要有`__init__.py`文件，别的文件才能引用之

```shell
developer@f50f43584025:~/webGame$ tree
.
|-- README.md
|-- db.sqlite3
|-- game	# 自建app项目
|   |-- __init__.py
|   |-- admin.py	# 管理员页面
|   |-- apps.py
|   |-- models.py	# 负责与数据库交互
|   |-- views.py	# 视图
|   |-- urls.py		# 路由
|   |-- templates	# 管理html文件
|   |-- static		# 静态文件
|   |   `-- __init__.py
|   |-- migrations
|   |   `-- __init__.py
|-- manage.py
|-- webGame	# 主项目
    |-- __init__.py
    |-- __pycache__	# 缓冲字节码文件
    |   |-- __init__.cpython-38.pyc
    |   |-- settings.cpython-38.pyc
    |   |-- urls.cpython-38.pyc
    |   `-- wsgi.cpython-38.pyc
    |-- asgi.py
    |-- settings.py
    |-- urls.py
    `-- wsgi.py
```

<br>

### 项目配置

- 配置文件在`project/project`目录下的`setting.py`内

- 如果访问自己的服务器的django项目，需要将服务器的ip地址填写到`ALLOWED_HOST`中

- 更改时区`TIME_ZONE`为国内的时区`Asia/Shanghai`

- 将自己新建的app加入配置到项目中，`INSTALLED_APPS`，存放的方式为`'APP_NAME.apps.' + apps.py内类名` 

- 配置静态文件路径

  - ```python
    STATIC_URL = '/static/'
    STATIC_ROOT = os.path.join(BASE_DIR, 'static')
    
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
    MEDIA_URL = '/media/'
    ```

- 连接mysql数据库

  - `pip install mysqlclient`

  - ```python
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql', # 连接 MySQL 数据库
            'NAME': 'django', # MySQL中数据库名称
            'USER': 'root',
            'PASSWORD': '123456',
            'HOST': 'localhost',
            'PORT': '3306',
        }
    }
    ```

- 添加模版路径

  - ```python
    TEMPLATES = [
        {
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [os.path.join(BASE_DIR, 'templates')],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.debug',
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                    'django.template.context_processors.static'
                ],
            },
        },
    ]
    ```

<br>

### 路由

作为网址和视图间的映射，将分析网址并提供相应的视图服务

```python
# 所有网址与函数的映射将写在urlpatterns内部

urlpatterns = [
    path('网址', 前往视图的哪个函数)
]
```

<br>

### 视图

也就是逻辑函数，对于用户的请求需要做出何种响应，直接在文件内定义函数即可

是Django的核心，负责接收请求，获取数据，返回结果

<br>

### 模型

> orm

django具有内在的账户系统可以满足简单的增删改查的用户需求，但是使用模型并不简单：

```shell
# 在model中编写模型类型
models
|--player
|--|--player.py
|--|--__init__.py
|--__init__.py

# 在app下的admin.py注册新表
admin.site.register(Player)

# 整合资源
python3 manage.py makemigrations
python3 manage.py migrate
```

模型类的编写，models.py

```python
from django.db import models


# Create your models here.
class BookInfo(models.Model):
    # 设置字段
    btitle = models.CharField(max_length=20)
    # 这里的信息将作为该记录的标题
    def __str__(self):
        return self.btitle

    
class HeroInfo(models.Model):
    # 设置字段
    hname = models.CharField(max_length=20)
    hgender = models.BooleanField()
    # 关联另一个表
    hbook = models.ForeignKey(BookInfo, on_delete=models.CASCADE)
    def __str__(self):
        return self.hname
    # 用于在管理员页面显示男女信息而非布尔值
    def gender(self):
        if self.hgender:
            return '男'
        else:
            return '女'
    gender.short_description = '性别'
```

在输入数据库变更命令后，将自动在mysql中建表（反向工程）

<br>

## Model

`from booktest.models import BookInfo, HeroInfo`

### Create

```python
# 创建对象
b = BookInfo()
# 给字段赋值
b.title = 'My Brother'
# 保存记录
b.save()

h = HeroInfo()
h.hname = '郭静'
h.hgender = True
# 表关联
h.hbook = b
h.save()
```

<br>

### Retrieve

```python
# 查询所有记录
BookInfo.objects.all()

# 升序排序
BookInfo.book1.all().order_by('bread')
# 降序排序
BookInfo.book1.all().order_by('-bread')

# 把对象中的属性构造成字典，返回列表
BookInfo.book1.all().values()

# 返回查询的总条数
BookInfo.book1.all().count()

# 判断查询结果中是否存在数据，避免出现逻辑错误
BookInfo.book1.filter(bread__lt = 20).exists()

# 判断某个字段是否非空
BookInfo.book1.filter(btitle__isnull = False)

# 判断是否在某个范围内
BookInfo.book1.filter(bread__in = [12, 16, 20, 24])

# 按照条件筛选唯一记录
# 如果筛选结果大于一条将会报错
b = BookInfo.objects.get(title = 'My Brother')

# 筛选满足条件的数据，多写几个filter等价“与”
BookInfo.book1.filter(bread__gt = 20).filter(bcomment__lt = 30)

# 过滤满足条件的数据
BookInfo.book1.exclude(bread__gt = 20)

# 获取与本表关联的记录
# 对象.关联类_set
b.heroinfo_set.all()

# 根据主表创建关联记录，可省略关联字段与save()
b.heroinfo_set.create(hname='欧阳疯', hgender=True)
```

#### 子查询运算符

属性 _ _ 比较运算符 = 值

- bread < 20     bread__lt = 20
- bread <= 20    bread__lte = 20
- bread > 20    bread__gt = 20
- bread >= 20   bread__gte = 20
- bread == 20    bread__exact = 20  或 bread = 20

#### 模糊匹配

```python
# 判断是否包含某个元素
BookInfo.book1.filter(btitle__contains = '传')
```

#### 跨表查询

模型类名_ _属性名_ _比较

```python
# 找出武功中含有“六”字的人所在的书
BookInfo.book1.filter(heroinfo__hcontent__contains = '六')
```

#### 聚合函数

需要导包`from django.db.models import *`

需要使用` aggregrate()`函数返回聚合函数中对应的值

- Avg()：求平均值
- Count()：总数
- Max()：最大值
- Min()：最小值
- Sum()：总和

```python
from django.db.models import Max
BookInfo.book1.all().aggregate(Max('bcomment'))
```

#### F & Q

##### F对象

一般情况下，字段都是写在等号的左边，如果要把字段写在等号的右边，需要使用 F 对象

比如：获取评论数大于阅读数的记录

```python
from django.db.models import F
BookInfo.book1.filter(bcomment__gt = F("bread"))
```

##### Q对象

如果要进行“或”操作，需要使用Q对象

比如：查询阅读数小于15或大于50的记录

```python
from django.db.models import Q
BookInfo.book1.filter(Q(bread__lt = 15) | Q(bread__gt = 50))
```



<br>

### Update

```python
# 获取对象
b = BookInfo.objects.get(title = 'My Brother')
# 更改
b.titile = 'My Sister'
b.save()
```

<br>

### Delete

```python
# 获取对象
b = BookInfo.objects.get(title = 'My Sister')
# 删除
b.delete()
```

<br>

## Admin

app/admin.py

```python
from django.contrib import admin
# 导入模型类
from booktest.models import BookInfo, HeroInfo


# 用来给BookInfoAdmin关联
class HeroInfoInline(admin.TabularInline):
    model = HeroInfo # 对应哪个模型
    extra = 3 # 一次增加3条记录


# 自定义管理员页面显示的内容
class BookInfoAdmin(admin.ModelAdmin):
    # 显示的字段
    list_display = ['id', 'title']
    # 过滤字段，允许按照指定字段过滤记录
    list_filter = ['title']
    # 搜索字段，允许按照指定字段搜索记录
    search_fields = ['title']
    # 分页
    list_per_page = 3
    # 属性出现的先后顺序，fields 和 fieldsets 只能有一个
    # fields = ['title', 'id']
    # 属性分组
    fieldsets = [
        ('basic', {'fields':['id']}),
        ('more', {'fields':['title']}),
    ]
    # 关联HeroInfo
    inlines = [HeroInfoInline]

    
class HeroInfoAdmin(admin.ModelAdmin):
    # 显示的字段，这里的gender是函数而非字段名
    list_display = ['id', 'hname', 'gender', 'hbook']
    

# Register your models here.
admin.site.register(BookInfo, BookInfoAdmin) # 新增管理类
admin.site.register(HeroInfo, HeroInfoAdmin)
```

<br>

## Urls

路由是项目中一个非常重要的概念，对于不同的需求应提供不同的服务

用户的需求通过网址来“表达”

路由也可以理解成项目的入口，而主入口的信息其实配置在settings.py中

`ROOT_URLCONF = 'webGame.urls'`

当用户在 URL 地址栏上写入地址的时候（[http://127.0.0.1:8000/booktest/123](http://127.0.0.1:8080/booktest/)）

解析器会把前面的协议（http)、地址（127.0.0.1）、端口号（8000）全部去掉

最后只剩下后面的内容（booktest/123）

会对应 settings.py 中 `ROOT_URLCONF`，随后按照path或者正则的规则分配路径

### 总路由

project/urls.py

```python
from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('admin/', admin.site.urls),
    # 导入路由，路径为app.urls
    path('', include('booktest.urls')),
]
```

### 子路由

app/urls.py

```python
from django.urls import path
from . import views


# 子路由
urlpatterns = [
    # 路径为视图内的函数名
    path('', views.index)
]
```

### 正则

希望能够匹配更多的路由

path(...)只能匹配固定的路径，不能匹配模糊的路径

如果想要匹配比较模糊的路径，可以考虑使用正则表达式

把 path 改为 url

```python
from . import views
from django.conf.urls import url


urlpatterns = [
    path('', views.index),
    # 获取所有字符串作为第一个参数发送给view.index函数
    url('^(.+)$', views.index),
    # 获取按照'_'分割的三个数字传给view.list函数
    url('^(\d+)_(\d+)_(\d+)$', views.goodlist, name='list')
]
```

### 命名空间

给某个路由命名，这样当后期业务发生改变时不至于`ag`整个项目一个一个改

project/urls.py

```python
urlpatterns = [
    path('', include(('booktest.urls', 'booktest'), namespace='booktest'))
]
```

app/urls.py

```python
urlpatterns = [
    url('^hero$', views.hero, name='hero'),
]
```

index.html

```html
<a href="{% url 'booktest:hero' %}">点击链接可正常访问</a>
```



<br>

## Views

项目提供的服务基本都在视图中完成，因此可以将视图按照不同的功能细分

```
developer@0ea578fb5d8b:~/webGame/game/views$ tree
.
├── __init__.py
├── index.py
├── menu
│   └── __init__.py
├── playground
│   ├── __init__.py
│   └── setScore.py
└── settings
    ├── __init__.py
    ├── acwing
    │   ├── __init__.py
    │   └── web
    │       ├── __init__.py
    │       ├── apply_code.py
    │       └── receive_code.py
    ├── getinfo.py
    ├── login.py
    ├── logout.py
    ├── ranklist.py
    └── register.py

```

```python
from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
def index(request):
    # HttpResponse直接将字符串返回到用户的页面上
    return HttpResponse('ok')
```

<br>

### Request

> request对象是视图函数的第一个参数

当服务器收到 http 协议请求后，创建 HttpRequest 对象

#### 属性

请求是用户发起的，所以 request 对象中大部分属性都是只读的

常见的属性：

- path：请求的路径（/booktest/）

- method：请求的HTTP方法（常见的是 GET 或 POST）

- encoding：提交的数据的编码方式

- - 如果为 None 表示使用浏览器默认的编码方式。一般都是utf-8
  - 这个属性是可以改写的，通过修改属性来修改编码方式（不过一般很少改写）

- GET：（全大写）是QueryDict对象（包装类），包含 get 请求中所有的参数

- POST：（全大写）是QueryDict对象（包装类），包含 post 请求中所有的参数

- FILES：（全大写）是MultiValueDict对象（包装类），包含上传文件的信息

- COOKIES：（全大写）包含全部的COOKIE信息

<br>

#### QueryDict对象

QueryDict是一个字典对象

request中的GET、POST都是 QueryDict 对象

注意 QueryDict 跟 dict 不同

dict是 python 的基本类型，QueryDict是封装类

QueryDict 可以处理一个键对应多个值

通过 get() 方法 可以根据键获取值，但是只能获取到最后一个值

提取 QueryDict 中的元素的方法

- dict.get(key, default)
- dict[key]

通过 getlist() 可以根据键获取所有的值，并以列表的形式返回

- dict.getlist(key, default)

<br>

### Response对象

Request对象是浏览器提交，在 Django 框架中创建完成

Response对象则是由程序员在视图函数中主动创建的

最简单的 Response 对象的写法

```python
def index(request):
    return HttpResponse('hello')
```

如果要使用模版，则需要render

```python
def index2(request):
    context = {'title': 'hello2'}
    return render(request, 'booktest/index2.html', context)
```

<br>

#### 属性

Response对象的属性包括：

- content：返回的内容。是 html 中的 body 的部分
- charset：响应的字符集。默认是 utf-8
- status_code：响应的 HTTP 状态码（200、404、403、407、500）
- content-type：输出的MIME类型（多媒体类型），通过报文来查看

<br>

#### 方法

- `__init__`：初始化。实例化 HttpResponse 对象
  - `response = HttpResponse()`

- write(content)：以文件的方式写，写入缓存中

- flush()：把缓存中的内容输出到页面

- set_cookie(key, value, max_age, expires)：设置cookie

- delete_cookie(key)：根据key删除对应的cookie

<br>

### HttpResponseRedirect

服务端的重定向跳转

浏览器的重定向跳转为`location.href = "http://xx.xx.xx"`

<br>

### JsonResponse

如果要返回 json 数据，当发起ajax请求的时候，返回json

```python
def jsonTest(request):
    data = {
        'name':'zhangsan',
        'age':23,
        'height':172.3,
        'married':True
    }
    return JsonResponse(data)
```

<br>

#### Session

***

##### session跟cookie的区别

session是服务器端的缓存；cookie是浏览器端的缓存

如果使用 cookie，所有的重要的信息都保存到浏览器端（客户端），这样会导致笔记重要的信息外泄

所以考虑把一些比较重要的信息保存到服务器端（比如登录的用户名和密码等），在浏览器端保存 session_key

***

##### 使用session

原则上每个 HttpRequest对象中都有一个 session 属性

session本质上也是字典的操作，与 cookie 类似

session使用base64 编码，可以在[这里](https://base64.us/ )进行在线解码

常用方法：

- clear()：清除所有session
- get(key)：根据键获取对应的值
- request.session[key] = value：给session中某个键赋值
- del request.session[key]：删除某个session

***

##### session过期时间

set_expiry(value)：设置session的过期时间

- 如果value不写，两个星期后过期
- 正整数：在value秒没有活动后过期
- imedelta对象：在当前时间加上一个值后过期
- 0：当浏览器关闭后过期
- None：永远不会过期

***

##### session存储位置

settings.py

```python
# session保存到数据库中 根据上面DATABASES的值来指定 默认
SESSION_ENGINE = 'contrib.sessions.backends.db'
# session保存到缓存中 速度比数据库要快 但一旦被清空 无法找回
SESSION_ENGINE = 'contrib.sessions.backends.cache'
# session保存到缓存和数据库中 先去缓存中查找 找不到再去数据库中查找
SESSION_ENGINE = 'contrib.sessions.backends.cache_db'
```



<br>

## Templates

> 在 Django中，模板使用的语法称为DTL（Django Templates Language）

DTL语法包含及部分：

1. 变量。使用两个花括号包含起来 {{ 变量名 }}
2. 标签。代码块 {% 代码块 %}
   1. 循环判断
   2. 加载外部信息
3. 过滤器。竖线(|)。{% 代码块 | 过滤函数 %}
4. 注释。{# 这里是注释 #}

<br>

### 传递参数

数据从视图传给模版，通过`render`的`context`参数

views.py

```python
def index(request, num):
    context = {
        'num': num
    }
    return render(request, 'index.html', context)
```

index.html

```html
<body>
   <!-- 通过{{}}从context字典中获取参数 -->
   {{num}}
</body>
```

<br>

### 流程控制

```html
<!-- 循环 -->
{% for ... in ... %}
{{ forloop.counter }} 表示当前循环的次数
{% empty %}
数据为空的时候显示的内容
{% endfor %}

<!-- 判断 -->
{% if %}
逻辑1
{% elif %}
逻辑2
{% else %}
逻辑3
{% endif %}
```

<br>

### 外部信息

- url
  - 反向地址解释
  - `{% url name p1 p2 %}`
  - name为命名空间
- static
  - 加载静态文件
  - `{% static '路径' %}`
  - 也可以`{{STATIC_URL}}{{路径变量}}`

<br>

### 过滤器

```python
# 直接使用
{{ 变量 | 过滤器 }}
{{name | lower | upper}}

# 搭配使用
{% if forloop.counter | divisibleby: "2" %}

# 给变量设置默认值
{{num | default:"0"}}

# 变量减一
{{num | add:-1}}
```

<br>

### 模版继承

把多个页面的公用部分抽取处理，做出父模板，并在子模板中继承父模板，然后再实现各自差异的功能

block标签。描述差异化，在父模板中预留某个区域，子模板中进行填充

base.html

```html
<!-- 父模版 -->
<html lang="en">
<head>
    ......
</head>
<body>
    <h1>top</h1>
    {% block block_name %}
        这里可以定义默认值，如果不定义默认值，则表示默认为空
    {% endblock %}
    <h1>bottom</h1>
</body>
</html>
```

注意：父模板需要定义整个页面的标签，也是唯一一个需要定义外部标签（html、head、body等）的文件

子模板不需要定义外部标签

index.html

```html
<!-- 路径为根目录为templates的相对路径 -->
{% extends 'base.html' %}

{% block block_name %}
实现填充内容
{% endblock %}
```

<br>

### HTML转义

如果直接把标签语言放到context传到前端通过dlt变量输出的时候浏览器无法识别

因此需要在前端进行转义

```html
<!-- 单行转移 -->
{{text | safe}}

<!-- 多行转义 -->
{% autoescape off %}
{{text}}
{{text}}
{{text}}
{% endautoescape %}
```



<br>

### 错误视图

- 404
  - 当访问一个不存在的网页的时候，会出现错误信息
- 500
  - 服务器内部错误

当项目上线后，debug=false，可以自定义错误视图，丰富用户体验

在settings.py的TEMPLATES内的DIRS中的路径下编写404.html，500.html即可解决这个问题

<br>

## 中间件

### CSRF

避免跨域请求伪造

在settings.py中MIDDLEWARE中注册

在使用post请求时必须通过csrf

在通过表单提交post请求时只需要加上`{% csrf_token %}`即可

```html
<form action="/booktest/csrf2" method="post">
    {% csrf_token %}
    <input type="text" name="name"/>
    <input type="submit" name="提交"/>
</form>
```

<br>

## 验证码

验证码的作用：防爬虫，减轻服务器的压力，防 CSRF

原理：通过某个页面，返回或展示一张图片，随机生成几个文字，登录的人员需要输入文字的内容，与图片一致才能完成匹配

安装插件：`pip install Pillow`

准备字体文件：`FreeMono.ttf`

生成验证码：

```python
from PIL import Image, ImageDraw, ImageFont
import random
def verifyCode(request):
    # 指定画布的高和宽(4个验证码 每个验证大小25*25)
    width = 100
    height = 25
    # 指定背景色
    bgColor = (64, 64, 64)
    # 创建画布
    # 参数1 mode 模型
    # 参数2 size 画布的背景色 是一个长度为2的元组
    # 参数3 color 画布的背景色 是一个长度为3的元组 RGB
    image = Image.new('RGB', (width, height), bgColor)
    # 创建画笔
    # 参数1 im 画布对象
    # 参数2 mode 模型 默认为None
    draw = ImageDraw.Draw(image)
    # 创建字体
    # 参数1 font 字体文件
    # 参数2 fontSize 字体大小
    font = ImageFont.truetype('FreeMono.ttf', 24)
    # 待显示的文字
    text = '0123456789ABCDEF'
    # 在画布上逐个描述字符
    textTemp = ''
    for i in range(4):
        textTemp1 = text[random.randrange(0, len(text))] # 0 1 2 3
        textTemp += textTemp1
        # 使用画笔在画布上绘制字符
        # 参数1 xy 要描述的字符左上角的坐标
        # 参数2 text 要描述的字符
        # 参数3 fill 要描述的字符的颜色 是一个长度为3的元组 RGB
        # 参数4 font 字体
        draw.text((25*i, 0), textTemp1, (255, 255, 255), font)

    print(textTemp)
    # 把生成的验证码保存到 session 中
    request.session['code'] = textTemp
    # 把画布保存到内存中
    import io
    buf = io.BytesIO()
    image.save(buf, 'png')

    return HttpResponse(buf.getvalue(), 'image/png')

```

验证验证码：

```python
def verifyTest2(request):
    sessionCode = request.session['code']
    postCode = None
    if 'verifyCode' in request.POST:
        postCode = request.POST['verifyCode']

    if postCode == sessionCode:
        return HttpResponse('ok')
    else:
        return HttpResponse('fail')

```

<br>

## 上传文件

在表单中设置enctype格式并通过input标签上传图片

```html
<body>
    <form action="/booktest/uploadHandler" method="post" enctype="multipart/form-data">
        {% csrf_token %}
        <input type="file" name="pic"/>
        <br>
        <input type="submit" value="上传"/>
    </form>
</body>
```

上传的文件将会被放在media目录下

```python
import os
from django5 import settings
def uploadHandler(request):
    files = request.FILES
    if 'pic' in files:
        pic1 = files['pic']
        picName = os.path.join(settings.MEDIA_ROOT, pic1.name)
        with open(picName, 'wb+') as pic:
            for c in pic1.chunks():
                pic.write(c)
        return HttpResponse('<img src="/static/media/%s">' % pic1.name)

    return HttpResponse('none')
```

<br>

## 分页

### Paginator

> 分页对象

Paginator(object_list, per_page)

参数1：列表数据

参数2：每页最多显示多少个数据

返回分页对象

#### 属性

- count：对象总数
- num_pages：页的总数
- page_range：页码范围。从第1页到第N页

<br>

### Pager

通过 Paginator 的 page() 方法返回 Pager 对象

#### 属性

- object_list：这一列上所有的序列
- number：当前页的序号，从1开始
- paginator：当前页所关联的 Paginator 的对象

#### 方法

- has_next()：是否有下一页
- has_previous()：是否有上一页
- has_other_pages()：是否有其他页，如果只有1页则返回False，如果有多页则返回True
- next_page_number()：下一页的页码
- previous_page_number()：上一页的页码
- len()：当前页中对象的个数

<br>

urls.py

```python
urlpatterns = [
    url('^heroList/(\d*)$', views.heroList, name='heroList'),
]
```

views.py

```python
from booktest.models import BookInfo, HeroInfo
from django.core.paginator import *


def heroList(request, pindex):
    if pindex == '':
        pindex = 1
    else:
        pindex = int(pindex)
    heroList = HeroInfo.objects.all()
    # Paginator对象
    paginator = Paginator(heroList, 5) # 每页显示5个数据
    # Page对象
    page = paginator.page(pindex)
    # 所有页的id
    pindexlist = paginator.page_range
    
    # 显示当前页page
    context = {
        'page': page,
        'pindexlist': pindexlist,
        'pindex': pindex,
    }
    return render(request, 'booktest/heroList.html', context)
```

booktest/heroList.html

```html
<body>
    <ul>
        {% for hero in page %}
        <li>{{ hero.hname }}</li>
        {% endfor %}
    </ul>
    <div class="pagenation">
   {% if pageid > 1 %}
             <a href=" "><上一页</a >
             {% endif %}

             {% for pindex in pindexlist %}
             <a href="/user/order{{pindex}}" {% if pindex == pageid %} class="active" {% endif %}>{{pindex}}</a >
             {% endfor %}

             {% if pageid < pindexlist|length %}
             <a href="/user/order{{pageid|add:1}}">下一页></a >
             {% endif %}
</div>
</body>
```

<br>

## Ajax

index.html

```html
<script type="text/javascript">
        $(function(){
            // 发送get请求
            $.get('/areatest/pro/', function(data){
                // 后端返回的数据
                console.log(data)
            });
            
            // 自定义请求
            $.ajax({
            url: "https://www.77zzl.top/settings/token/refresh/",
            type: "post",
            data: {
                refresh: this.root.refresh,
            },
            success: resp => {
                // 后端返回的数据
                console.log(resp)
            },
        })
        });
</script>
```

<br>

## 富文本编辑器

插件：[tinymce](https://pypi.org/project/django-tinymce/)

安装：`pip install django-tinymce`

配置：

- settings.py

  - ```python
    INSTALLED_APPS = [
        ......
        'tinymce'
    ]
    ```

- urls.py

  - ```python
    urlpatterns = [
        ......
        path('tinymce/', include(('tinymce.urls', 'tinymce'), namespace='tinymce')),
    ]
    ```

- models.py

  - ```python
    from django.db import models
    from tinymce.models import HTMLField
    
    
    class Test1(models.Model):
        content = HTMLField()
    ```

- 数据迁移

  - ```shell
    python manage.py makemigrations
    python manage.py migrate
    ```

- 场景一：在管理员页面中使用富文本编辑器

  - admin.py

  - ```python
    from django.contrib import admin
    from booktest.models import Test1
    
    
    # Register your models here.
    admin.site.register(Test1)
    ```

  - 随后只需要在管理页面新增记录即可看到富文本编辑器

- 场景二：在自定义页面中使用富文本编辑器

  - htmlEditor.html

  - ```html
    <head>
        <script src="/static/tinymce/tinymce.min.js"></script>
        <script type="text/javascript">
            tinyMCE.init({
                'selector': 'textarea', // 选择器，标签名
                'mode': "textareas",
                'theme': "silver",
                'plugins': "spellchecker,directionality,paste,searchreplace",
                'width': 600,
                'height': 400
            });
    	</script>
    </head>
    <body>
        <form action="/booktest/htmlEditorHandle" method="post">
            {% csrf_token %}
            <textarea name="hcontent"></textarea><br>
            <input type="submit" value="提交"/>
        </form>
    </body>
    ```

  - views.py

  - ```python
    def htmlEditorHandle(request):
        post = request.POST
        if 'hcontent' in post:
            html = post['hcontent']
            # 保存到数据库中
            #text1 = Test1.objects.get(pk = 1)
            #text1.content = html
            #text1.save()
            # 往数据库中添加
            text1 = Test1()
            text1.content = html
            text1.save()
            context = {'content': html}
            return render(request, 'booktest/htmlShow.html', context)
    ```

  - htmlShow.html

  - ```html
    <body>
        {{ content | safe }}
    </body>
    ```

<br>

## Cache

### 配置

- 启动redis-server
  - `sudo /etc/init.d/redis-server /etc/redis/redis.conf`
- 重启redis
  - `sudo /etc/init.d/redis-server restart`

- django中集成redis
  - `pip install django_redis`

settings.py

```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    },
}
USER_AGENTS_CACHE = 'default'
```

views.py

```python
from django.core.cache import cache


# 将state存入redis，有效期两小时
cache.set(state, True, 7200)

# 获取某个值
cache.get(state)

# 获取所有值
cache.keys()

# 模糊匹配
cache.keys('*%s*' % (self.uuid))
```

<br>

## Websocket

之前所有的操作都是在http协议上的，但是当我们需要建立长链接时，http则无法满足我们的需求

### 安装

```shell
pip install channels_redis
```

<br>

### 配置

project/asgi.py

```python
import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from game.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'acapp.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(URLRouter(websocket_urlpatterns))
})
```

project/settings.py

```python
INSTALLED_APPS = [ 
    ......
    'channels'
]

...

ASGI_APPLICATION = 'acapp.asgi.application'
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("127.0.0.1", 6379)],
        },
    },
}
```

<br>

### 路由

在应用目录下创建`routing.py`，起websocket中路由作用

```python
from django.urls import path
from game.consumers.multiplayer.index import MultiPlayer

websocket_urlpatterns = [
    path("wss/multiplayer/", MultiPlayer.as_asgi(), name="wss_multiplayer"),
]


作者：yxc
链接：https://www.acwing.com/blog/content/12692/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

注意websocket是异步请求，因此需要通过`as_asgi()`将其转为异步方法

<br>

### 视图

在应用目录下创建`consumer.py`，起视图作用（消费者）

这里用到一个消费者生产者的概念，比如说在一个多人聊天室内，当某个时刻A发送了一条信息，那么A就充当了**生产者**的身份将消息传给聊天室内其余的人，而其余的人便是**消费者**，整个聊天室便是**组**，消费者与生产者通信的媒介称为**管道**

```python
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class MultiPlayer(AsyncWebsocketConsumer):
    # 建立连接
    async def connect(self):
        # 后端同意连接则调用accept，表示建立连接成功
        await self.accept()
        print('accept')

        # 在管道内添加组
        self.room_name = "room"
        await self.channel_layer.group_add(self.room_name, self.channel_name)

    # 断开连接
    async def disconnect(self, close_code):
        # 在管道内移除组
        await self.channel_layer.group_discard(self.room_name, self.channel_name)

    # 将消息群发到组
    async def group_send_event(self, data):
        # 将dict类型的数据转成str
        await self.send(text_data=json.dumps(data))
        
	# 接收信息
    async def receive(self, text_data):
        # 将str类型的数据转换为dict类型
        data = json.loads(text_data)
        print(data)
        
        # 特别注意在异步中使用数据库的方式
        # 必须把查询语句封装成函数
        def db_get_player():
            return Player.objects.get(user__username=data['username'])
        # 再将函数变成异步
        player = await database_sync_to_async(db_get_player)()


作者：yxc
链接：https://www.acwing.com/blog/content/12692/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

<br>

### 启动

```
daphne -b 0.0.0.0 -p 5015 project.asgi:application
```

<br>

## JWT

> json web token

用来解决跨域身份验证问题，即前后端分离后身份验证的问题

### 身份验证

#### 传统

传统的身份验证是通过`session_id`来解决的

1. 客户端向服务端发送第一次请求
2. 服务端向客户端发送一段用来验证身份的`session_id`，并将其存在服务端的数据库中
3. 客户端接收到的`session_id`将被存放在浏览器的`cookie`中，此时客户端的`js`无法获取`cookie`
4. 当客户端再次向服务器发送请求是，浏览器自动带上`cookie`
5. 服务端接收到客户端的请求并从中读取`cookie`与自己数据库中的`cookie`进行匹配来进行身份验证

可以发现当前后端分离后，便出现了跨域问题，因为向服务端发送请求的前端不一定是浏览器，无法自动携带上`cookie`，为了解决这个问题，常见的处理方案便是`jwt`

#### 加密

1. 客户端向服务端发送第一次请求
2. 服务端向客户端发送一段用来验证身份的`token`和用户信息
   1. `token`是由用户信息和服务器内的密钥加密而成的一串公钥
   2. 用户信息与密钥加密的结果唯一
   3. 服务器内的密钥唯一且对客户端来说不可知，唯一表示的是所有公钥都使用同样的密钥解密
3. 客户端再次向服务器发送请求时将公钥放进请求头中
4. 服务器将用户信息和密钥加密并与请求头中的公钥进行匹配，匹配成功则表示身份验证通过

<br>

### django rest framework

django提供的可视化API插件，JWT需与该插件一起使用

在编写API时与之前有所不同，不再是定一个函数并在函数内实现业务逻辑，而是定义一个类并按照四种不同的请求命名函数来处理不同的业务逻辑

#### 配置

- 安装
  - `pip install djangorestframework`
  - `pip install pyjwt`
- 配置
  - `settings.py/INSTALL_APPS`中添加`rest_framework`

<br>

#### API编写

```python
from rest_framework.views import APIView
from rest_framework.response import Response

class SnippetDetail(APIView):
    def get(self, request):  # 查找
        ...
        return Response(...)

    def post(self, request):  # 创建
        ...
        return Response(...)

    def put(self, request, pk):  # 修改
        ...
        return Response(...)

    def delete(self, request, pk):  # 删除
        ...
        return Response(...)

作者：yxc
链接：https://www.acwing.com/blog/content/21139/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

<br>

### simplejwt

使用`DRF`实现的`JWT`，用来替代`csrf`验证方式

#### 配置

- 安装
  - `pip install djangorestframework-simplejwt`
- 配置
  - `settings.py/INSTALL_APPS`中添加`rest_framework`

在`settings.py`文末添加如下内容

```python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    )
}

from datetime import timedelta
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=5),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
    'ROTATE_REFRESH_TOKENS': False,
    'BLACKLIST_AFTER_ROTATION': False,
    'UPDATE_LAST_LOGIN': False,

    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'VERIFYING_KEY': None,
    'AUDIENCE': None,
    'ISSUER': None,
    'JWK_URL': None,
    'LEEWAY': 0,

    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'USER_AUTHENTICATION_RULE': 'rest_framework_simplejwt.authentication.default_user_authentication_rule',

    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
    'TOKEN_USER_CLASS': 'rest_framework_simplejwt.models.TokenUser',

    'JTI_CLAIM': 'jti',

    'SLIDING_TOKEN_REFRESH_EXP_CLAIM': 'refresh_exp',
    'SLIDING_TOKEN_LIFETIME': timedelta(minutes=5),
    'SLIDING_TOKEN_REFRESH_LIFETIME': timedelta(days=1),
}

作者：yxc
链接：https://www.acwing.com/blog/content/21140/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

如果要使用websocket还需要另外在app目录下创建文件`channelsmiddleware.py`

```python
"""General web socket middlewares
"""

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import UntypedToken
from rest_framework_simplejwt.authentication import JWTTokenUserAuthentication
from channels.middleware import BaseMiddleware
from channels.auth import AuthMiddlewareStack
from django.db import close_old_connections
from urllib.parse import parse_qs
from jwt import decode as jwt_decode
from django.conf import settings
@database_sync_to_async
def get_user(validated_token):
    try:
        user = get_user_model().objects.get(id=validated_token["user_id"])
        # return get_user_model().objects.get(id=toke_id)
        return user

    except:
        return AnonymousUser()



class JwtAuthMiddleware(BaseMiddleware):
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
       # Close old database connections to prevent usage of timed out connections
        close_old_connections()

        # Try to authenticate the user
        try:
            # Get the token
            token = parse_qs(scope["query_string"].decode("utf8"))["token"][0]

            # This will automatically validate the token and raise an error if token is invalid
            UntypedToken(token)
        except:
            # Token is invalid

            scope["user"] = AnonymousUser()
        else:
            #  Then token is valid, decode it
            decoded_data = jwt_decode(token, settings.SIMPLE_JWT["SIGNING_KEY"], algorithms=["HS256"])

            # Get the user using ID
            scope["user"] = await get_user(validated_token=decoded_data)
        return await super().__call__(scope, receive, send)


def JwtAuthMiddlewareStack(inner):
    return JwtAuthMiddleware(AuthMiddlewareStack(inner))

作者：yxc
链接：https://www.acwing.com/blog/content/21140/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

<br>

#### Access & Refresh

- access
  - jwt的核心，也就是令牌，默认有效期五分钟
- refresh
  - 用来刷新access，默认有效期十四天

<br>

##### 通过API获取

编写api路由

```python
# urls.py
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
```

前端获取api代码

```js
// 用refresh刷新access
$.ajax({
    url: "https://www.77zzl.top/settings/token/refresh/",
    type: "post",
    data: {
        refresh: this.root.refresh,
    },
    success: resp => {
        this.root.access = resp.access
        window.localStorage.setItem("access", resp.access)
    },
})

		// 获取access和refresh
		$.ajax({
            url: "https://www.77zzl.top/settings/token/",
            type: "post",
            data: {
                username: username,
                password: password
            },
            success: resp => {
                this.root.access = resp.access
                this.root.refresh = resp.refresh

                // 存入本地缓存
                storage.setItem("access" ,resp.access)
                storage.setItem("refresh", resp.refresh)
            },
        })

		// 注意对于需要jwt验证的api需要多加一个headers
        $.ajax({
            url:"https://www.77zzl.top/settings/getinfo/",
            type: "get",
            headers: {
                'Authorization': "Bearer " + this.root.access,
            },
            success: resp => {
                console.log('success')
            },
        })
```

```python
# views/getinfo.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from game.models.player.player import Player


class InfoView(APIView):
    # 如果需要jwt认证则加上
    permission_classes = ([IsAuthenticated])

    def get(self, request):
        # 可以直接从request获取数据，无需经过request.GET
        user = request.user

        # 必须用drf的Response不能用http里面的
        return Response({
            'result': "success",
        })
```

##### 第三方授权可以手动获取

```python
from rest_framework_simplejwt.tokens import RefreshToken

def get_tokens_for_user(user):
    # 手动获取
    refresh = RefreshToken.for_user(user)

    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }
```

<br>

## BUG

PS. 如果迁移时出现

`No migrations to apply`的提示

则需要进行如下操作：

1 删除该app名字下的migrations目录下的`__init__.py`等文件。但`migrations`目录需要保留，并创建一个新的空的`__init__.py`文件。

2 进入数据库，找到django_migrations的表，删除该app名字的所有记录。

3 `python manage.py makemigrations`

4 `python manage.py migrate`

原因是django_migrations表记录着数据库的对应表的修改记录

***

PS. 如果迁移时出现

`Table 'django_content_type' already exists`的提示

则需要进行如下操作：

1 `python manage.py migrate --fake`

2 `python manage.py migrate`

***

PS. 使用session时可能出现

`(1146, "Table 'django3.django session' doesn't exist")`

这是因为目前数据库中还没有 django_session 表

此时，需要做一个迁移

若没有模型类

只能进行空的迁移

```
python manage.py makemigrations --empty app
python manage.py migrate
```

***

[部署域名](https://www.acwing.com/file_system/file/content/whole/index/content/3419874/)

[AcWing一键登录，Oauth2](https://www.acwing.com/blog/content/12466/)
