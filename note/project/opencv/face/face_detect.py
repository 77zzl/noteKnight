import cv2 as cv


def face_detect_cir(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray,
                                           scaleFactor=1.01,
                                           minNeighbors=9,
                                           minSize=(29, 29),
                                           maxSize=(35, 35))

    for face in faces:
        x, y, w, h = face
        center = (x + w // 2, y + h // 2)
        radius = max(w, h)
        color = (0, 255, 255)  # 黄色=绿色+红色
        cv.circle(img, center, radius, color, thickness=2)


def face_detect_rec(img):
    # 彩色 512*512*3
    # 灰度 512*512
    # 一般都需要把彩色图片转灰度图片后进行图片识别和处理
    # 彩色图片图片转灰度图片
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    # 进行人脸检测
    # 返回一个个的矩形框。表示一个个人脸
    # 参数1 image 待检测的图片
    # 参数2 scaleFactor 缩放因子
    # 参数3 minNeighbors 最小近邻
    # 参数4 minSize 人脸方框边长最小值（长度为2的元组）
    # 参数5 maxSize 人脸方框边长最大值（长度为2的元组）
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
        # 绘制圆形怎么做？


img = cv.imread('./data/img/1.jpeg')
face_detect_rec(img)
# face_detect_cir(img)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.imwrite('./data/img/1_find.jpeg', img)
print(type(img))
cv.destroyAllWindows()

