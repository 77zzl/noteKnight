# ECharts

![](C:\Users\penny\Desktop\image\微信截图_20211127102828.png)

### 入门教程

***

[点击官网教程](https://echarts.apache.org/handbook/zh/get-started)

#### 首先列举可能用到的技术

1. springboot
2. thymeleaf
3. javascript



#### 其次是需要设置的配置

###### springboot的pom

```
<properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
        <mysql-connector-java.version>5.1.38</mysql-connector-java.version>
        <druid.version>1.1.6</druid.version>
    </properties>


    <!--继承父工程的starter-->
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.3.4.RELEASE</version>
    </parent>

    <dependencies>
        <!--springboot应用支持springmvc功能-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- mysql driver -->
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>${mysql-connector-java.version}</version>
        </dependency>

        <!--数据源连接池-->
        <dependency>
            <groupId>com.alibaba</groupId>
            <artifactId>druid</artifactId>
            <version>${druid.version}</version>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-configuration-processor</artifactId>
            <optional>true</optional>
        </dependency>

        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>

        <!--视图组件：thymeleaf-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-thymeleaf</artifactId>
        </dependency>

        <dependency>
            <groupId>com.alibaba</groupId>
            <artifactId>fastjson</artifactId>
            <version>1.2.14</version>
        </dependency>

    </dependencies>
```



###### 最好把yml文件也写上

```
server:
  port: 8083
```



###### 文件目录的约定

![](C:\Users\penny\Desktop\image\微信截图_20211127103416.png)



### Learn

***

#### Springboot

最基础的一点则是springboot的基础知识，运用在echarts可视化上的技术归纳如下

1. 各目录的作用
2. 注解的作用



###### 各目录的作用

```
jave
----com.zhang						# 包名
	----MainApplication.java		# 启动类，内置main函数
	----controller
		----Controller				# 控制器，用于页面跳转的导航相当于django的url
resources
----static							# 静态文件
	----js
----templates
	----html
----application.yml					# 配置文件
```



###### 注解的作用

注解可以理解为声明本文件在框架中的作用

启动类用到的注解`@SpringBootApplication`，顾名思义，即为整个框架的main函数所在

控制器的注解`@Controller`，同样用来声明本文件用于控制url的路径导航



#### thymeleaf

本案例中队thymeleaf的使用比较基础，可归纳为以下几点

1. 首先在html中指出xmlns `<html lang="en" xmlns:th="http://www.thymeleaf.org">`

2. html中的script代码即可调用controller中的变量，做法如下

3. 在控制器中将变量塞进视图组件 `modelMap.put("dataJson",dataJson);`

4. 在javascript代码中接收变量 `var data=[[${dataJson}]];`

   

#### javascript

本案例中将javascript代码放进了html中，而且导入了案例所需的echarts.js包，看看这些功能是如何实现的

在html的head中添加 `<script src="/js/echarts.js"></script>`，便可将echarts.js包导入

在html中使用javascript语言 `<script type="text/javascript" th:inline="javascript"></script>`，便可在标签中直接写入javascript语法



### Run

***

###### Controller

```java
// 在网址后写上mapping的路径则会在服务器中访问该程序
@RequestMapping("/doCharts")
	// 注意要将数据放进ModelMap视图组件中才可在前端显示
    public String doCharts(ModelMap modelMap)
    {
        // 定义商品名
        List<String> data1=new ArrayList<>();
        data1.add("手机");
        data1.add("电脑");
        data1.add("平板");

        // 定义销售量
        List<Integer> data2=new ArrayList<>();
        data2.add(10);
        data2.add(20);
        data2.add(30);

        // 将数据转换成json数据格式
        String data1Json= JSON.toJSONString(data1);
        String data2Json= JSON.toJSONString(data2);

        //将数据转发成视图组件显示
        modelMap.put("data1Json",data1Json);
        modelMap.put("data2Json",data2Json);

        // 访问html页面，程序将自动访问resources/templates下的同名文件
        return "echartsdemo";
    }
```



###### html

```html
<!DOCTYPE html>
<html lang="en" xmlns:th="http://www.thymeleaf.org">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <script src="/js/echarts.js"></script>
</head>
<body>

<div id="main" style="width: 600px;height: 400px;"></div>
<script type="text/javascript" th:inline="javascript">

    // 基于准备好的dom，初始化echarts实例，'main'是本div的id
    var myChart = echarts.init(document.getElementById('main'));

    // 在js读取thymeleaf变量值
    var data1=[[${data1Json}]];
    var data2=[[${data2Json}]];

    // 指定图表的配置项和数据
    var option = {
        title: {
            text: 'ECharts 入门示例'
        },
        tooltip: {},
        legend: {
            // 什么的数据
            data: ['销量']
        },
        xAxis: {
            // x轴上的数据
            data: JSON.parse(data1)
        },
        yAxis: {},
        // 具体数据信息
        series: [
            {
                name: '销量',
                type: 'bar',
                data: JSON.parse(data2)
            }
        ]
    };

    // 使用刚指定的配置项和数据显示图表。
    myChart.setOption(option);
</script>
</body>
</html>
```

