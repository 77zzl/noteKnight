# 响应式布局

根据用户屏幕逻辑大小展现不同的布局。

<br>

## Media

在css中根据屏幕大小手写不同的样式

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="./css/responsive.css">
    <title>Document</title>
</head>

<body>
    <div class="container">
        <div class="card"></div>
    </div>
</body>

</html>
```

```css
/* 默认样式 */
.container {
    background-color: black;
    width: 80%;
    /* 上下居中
        左右居中 margin: auto;
    */
    margin: 0 auto;
    padding: 10px;
}

.card {
    width: 80%;
    height: 100vh;
    background-color: aqua;
    margin: 0 auto;
}

/* 根据屏幕大小调整样式 */
/* 当屏幕宽度尺寸小于768px时使用该media内定义的样式 */
@media (min-width: 768px) {
    .card {
        background-color: blue;
    }
}

/* 当屏幕宽度尺寸小于992px时使用该media内定义的样式 */
@media (min-width: 992px) {
    .card {
        background-color: brown;
    }
}
```

<br>

## 栅栏布局