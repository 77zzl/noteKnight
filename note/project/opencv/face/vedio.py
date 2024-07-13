import cv2 as cv


def face_detect_demo(img):
    # 彩色 512*512*3
    # 灰度 512*512
    # 一般都需要把彩色图片转灰度图片后进行图片识别和处理
    # 彩色图片图片转灰度图片
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    # 进行人脸检测
    # 返回一个个的矩形框。表示一个个人脸
    faces = face_detector.detectMultiScale(gray)
    # 遍历每个人脸
    for face in faces:
        x, y, w, h = face
        # 绘制矩形
        pt1 = (x, y)
        pt2 = (x+w, y+h)
        color = (0, 0, 255)
        cv.rectangle(img, pt1, pt2, color, thickness=2)
        # 绘制圆形怎么做？


if __name__ == '__main__':
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

        ret = cv.waitKey(100) # 0.01秒
        if ord('q') == ret:
            break

    cv.destroyAllWindows()
    # 释放视频的内存空间
    cap.release()

