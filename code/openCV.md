### 回收资源

```python
# 只要使用了opencv必须回收资源
cv.destroyAllWindows()
```

<br>

### 导入和导出图片

```python
# 导入图片
img = cv.imread('./data/img/face.jpg')

# 弹窗显示图片
# 参数一：图片名称
# 参数二：图片的numpy对象
cv.imshow('img', img)
# 只有规定了等待时间才能看到图片，否则一闪而过
cv.waitKey(0)

# 导出图片
# 参数一：图片路径
# 参数二：图片的numpy对象
cv.imwrite('img_new.jpg', img)
```

<br>

### 灰度图

灰度图不影响识别人脸但是可以减少计算量

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

<br>

### 改变大小

```python
resize_img = cv.resize(img, dsize=(400, 300))
```

<br>

### 绘制图形

##### 矩形

参数1：img。在哪个目标上绘制矩形

参数2：pt1。矩形左上角的点。长度为2的元组

参数3：pt2。矩形右下角的点。长度为2的元组

参数4：color。矩形的颜色。长度为3的元组（BGR）

参数5：thickness。矩形线条的厚度。默认为2

```python
cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
```

##### 圆形

参数1：img。在哪个目标上绘制圆形

参数2：center。圆心的点。长度为2的元组

参数3：radius。圆的半径。一个值

参数4：color。矩形的颜色。长度为3的元组（BGR）

参数5：thickness。矩形线条的厚度。默认为2

```python
cv.circle(img, (x, y), 50, (0, 255, 255), thickness=2)
```

<br>

### 人脸检测

##### 预处理

- 调整为灰度

##### 特征数据

- 加载特征数据

##### 人脸检测

- 使用特征数据进行人脸检测
  - 参数1 image 待检测的图片
  - 参数2 scaleFactor 缩放因子
  - 参数3 minNeighbors 最小近邻
  - 参数4 minSize 人脸方框边长最小值（长度为2的元组）
  - 参数5 maxSize 人脸方框边长最大值（长度为2的元组）

##### 绘制图形

- 矩形
- 圆形

```python
def face_detect_rec(img):
    # 灰度
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    # 进行人脸检测
    faces = face_detector.detectMultiScale(gray,
                                           scaleFactor=1.01,
                                           minNeighbors=3,
                                           minSize=(29, 29),
                                           maxSize=(35, 35))
    # 遍历每个人脸
    for face in faces:
        x, y, w, h = face
        
        # 绘制矩形
        pt1 = (x, y)
        pt2 = (x+w, y+h)
        color = (0, 0, 255)
        cv.rectangle(img, pt1, pt2, color, thickness=2)
        # 绘制圆形
        center = (x + w // 2, y + h // 2)
        radius = max(w, h)
        color = (0, 255, 255)  # 黄色=绿色+红色
        cv.circle(img, center, radius, color, thickness=2)
```

<br>

### 人脸检测—视频

```python
# 读取视频
    cap = cv.VideoCapture('./data/video/qiaolan.mp4')
    while True:
        # 读取一帧
        # 返回值1。flag：如果视频一直在播放，返回True。表示还有下一帧（如果是打开摄像头，则该值永远为True）
        # 返回值2。frame：对于的图片帧

        flag, frame = cap.read()
        if not flag:
            break

        face_detect_demo(frame)
        cv.imshow('img', frame)

        ret = cv.waitKey(10) # 0.01秒
        if ord('q') == ret:
            break

    cv.destroyAllWindows()
    # 释放视频的内存空间
    cap.release()
```

<br>

### 人脸识别

#### 训练

- 准备特征数据、对应的标签
- 准备检测的工具
- 导入图片
- 将图片对应的标签放进标签列表
- 对数据进行预处理
- 人脸识别
- 将识别出来的人脸矩阵放进特征数据列表
- 生成人脸识别器实例模型
- 训练模型
- 导出模型

```python
import cv2 as cv
import os
import numpy as np


'''
训练模型需要的两个参数，特征+标签
参数1:numpy列表，人脸的特征数据
参数2:整形列表，每个特征的标签
'''
def readIMage(path):
    # 特征、标签
    faceSamples, ids = [], []
    # 图片路径列表
    imagePaths = []
    for f in os.listdir(path):
        fullpath = path + f
        imagePaths.append(fullpath)

    # 检测人脸的工具
    face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

    for imagePath in imagePaths:
        id = os.path.split(imagePath)[-1].split('.')[0].split('_')[0]
        # 导入数据
        img = cv.imread(imagePath)
        # 预处理
        # resized_img = cv.resize(img, (92, 112))
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 人脸识别
        faces = face_detector.detectMultiScale(gray_img)
        for face in faces:
            x, y, w, h = face
            faceImg = gray_img[y:y+h, x:x+w]
            faceSamples.append(faceImg)
            ids.append(int(id))

    return faceSamples, ids


if __name__ == '__main__':
    path = './data/train/'
    faceSamples, ids = readIMage(path)
    # 生成人脸识别器实例模型
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # 训练模型
    recognizer.train(faceSamples, np.array(ids))
    # 导出模型
    recognizer.write('./data/trainer.yml')
```

<br>

#### 预测

- 导入模型
- 导入待预测的图片
- 检测人脸
- 使用模型识别检测出来的人脸判断置信度

```python
import cv2 as cv


# 获取训练数据文件
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('./data/trainer.yml')
# 要预测的文件
predict_file = './data/test/8.JPG'
img = cv.imread(predict_file)
# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 检测人脸的工具
face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
# 进行人脸检测
faces = face_detector.detectMultiScale(gray)
for face in faces:
    x, y, w, h = face
    faceImg = gray[y:y+h, x:x+w]
    # 预测
    id, confident = recognizer.predict(faceImg)
    print('这是第%d个人，置信度为%f' % (id, confident))
```

