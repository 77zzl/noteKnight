# HTML



## 文件结构

#### HTML的整体框架

```html
<!-- 按!回车生成基础框架 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    
</body>
</html>
```

- 可以看到 `html` 包含了 `head` 和 `body`



#### 语义标签

语义标签很好的划分了整个页面的不同结构，帮则开发者快速构建网页

- header
  - 用于展示介绍性的内容
    - logo、搜索框、作者名
- nav
  - 提供导航链接
    - 菜单、目录和索引
- figure
  - 引用图片表格或代码块时使用
- figcaption
  - 对图片等引用信息加以说明或标题
- article
  - 表示页面中的独立结构
    - 文章、博客、评论
- aside
  - 一个和其余页面几乎无关的部分
- footer
  - 页脚

***

从语义标签中不难理解到，一个基础页面可以用不同的标签加以区分，但在实际中并非一定要使用上述标签进行页面布局，应以完成项目预期为最高目标节省时间开销



## 标签

所有标签都可以分为以下两类

- div
  - 块级标签——独占一行会换行
- span
  - 内联标签——不换行



#### 文本标签

```html
<h1>
    共有六个级别表示不同级别的标题
</h1>

<p>
    一个段落，会自动换行且段间留白
</p>

<pre>预定义格式文本并无特别的样式，内容将直接呈现</pre>

<br>
换行

<hr>
一条横线罢了

<i>斜体</i>
<b>加粗</b>
<del>删除线</del>
<ins>下划线</ins>
```



#### 图片

```html
<img src="" alt="" height="" width="">
```

- src
  - 图片地址
- alt
  - 文本描述
- height
  - 高度
- width
  - 宽度



#### 超链接

```html
<a href="" target="_blank"></a>
```

- href
  - 表示链接地址
- target
  - 参数为"_blank"时表示打开新标签页面



#### 表单

```html
<!-- action表示要发生的交互行为 -->
<form action="">
        <!-- type内有多种可选参数
            text 文本
            number 数字
            email 邮箱（将自动检查是否为正确的邮箱格式）
            password 密码
            radio 复选框 
        -->
        <input type="text">
</form>
```

input内的属性不止这些：

- name
  - 用于表单提交时
- id
  - 用于样式，这也是每个标签都可以有的属性
- maxlength & minlength
  - 最大最小长度
- required
  - 是否必填
- placeholder
  - 当表单控件为空时显示的内容



> 其他标签

```html
<textarea>多行纯文本编辑控件</textarea>

<!-- 复选框
	label表示问题
	option内的是可选的回答
-->
<label for="pet-select">Choose a pet:</label>
<select name="pets" id="pet-select">
    <!-- 复选框再未选择时会呈现第一个选项的文本 -->
    <option value="">--Please choose an option--</option>
    <option value="dog">Dog</option>
    <option value="cat">Cat</option>
    <option value="hamster">Hamster</option>
    <option value="parrot">Parrot</option>
    <option value="spider">Spider</option>
    <option value="goldfish">Goldfish</option>
</select>

<button>
    一个可点击的按钮
</button>
```



#### 特殊符号

```html
<!-- < -->
&lt;

<!-- > -->
&gt;

<!-- & -->
&amp;

<!-- 不断行的空白 -->
&nbsp;
```





# CSS

写在 `<head>` 内的标签

- meta
  - charset
    - 字符编码
  - link
    - <link rel="icon" href="images/icon.png">
      - 网页标题图片



## 样式的定义方式

- 行内定义

  - <a href="" target="_blank"></a>

- 内部定义

  - <style>
        img {
            width: "";
            height: "";
            border-radius: "";
        }
    </style>

- 外部样式
  - <link rel="stylesheet" href="">



## 选择器

#### 标签选择器

```css
div {
    width: 200px;
    height: 200px;
    background-color: gray;
}
```

所有的div都会使用这些样式

`<div><div>`



#### ID选择器

```css
#gec007 {
    width: 200px;
    height: 200px;
    background-color: gray;
}
```

因为 `id` 是唯一的，因此仅有某一个为该 `id`  的标签会作用这个选择器

`<div id = "gec007"><div>`



#### 类选择器

```css
.rectangle {
    width: 200px;
    height: 200px;
    background-color: gray;
}
```

`<div class = "rectangle"><div>`



#### 伪类选择器

- :link：链接访问前的样式
- :visited：链接访问后的样式
- :hover：鼠标悬停时的样式
- :active：鼠标点击后长按时的样式
- :focus：聚焦后的样式



#### 复合选择器

- element1, element2：同时选择元素element1和元素element2。
- element.class：选则包含某类的element元素。
- element1 + element2：选择紧跟element1的element2元素。
- element1 element2：选择element1内的所有element2元素。
- element1 > element2：选择父标签是element1的所有element2元素。





























































