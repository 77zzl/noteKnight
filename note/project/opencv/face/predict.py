import cv2 as cv


# 获取训练数据文件
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('./data/trainer_cat.yml')
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
