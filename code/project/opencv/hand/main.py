import cv2 as cv
import numpy as np
from handDetector import HandDetector


def main():
    # 保存图片的个数
    save_nums = 0
    # 打开摄像头，参数为0时表示打开内置摄像头
    cap = cv.VideoCapture(0)
    # 创建手势识别对象
    detector = HandDetector()
    # 手指头列表
    tip_idx = [4, 8, 12, 16, 20]
    # 手势图片路径
    fingersImgPath = [f'./data/fingers/{i}.png' for i in range(6)]
    # 手势图片导入
    fingersImgLst = [cv.imread(i) for i in fingersImgPath]

    while True:
        # 因为打开的是摄像头所以第一个参数永远为True
        _, img = cap.read()

        # 因为摄像头是镜像所以需要反转画面
        img = cv.flip(img, 1)

        # 绘制手势
        img = detector.find_hands(img)

        # 获取手势数据
        handslst = detector.find_positions(img)
        for i, hand in enumerate(handslst):
            # 在指尖描绘圆点
            fingers = []
            '''
            ！！！特别注意！！！
            掌心面向屏幕，掌背面向自己区分左右手
            左右手的判断根据: 食指掌根5 和 中指掌根9 位置进行判断

            大拇指要特殊处理
            首先是根据左右手区别处理
            根据 大拇指指尖4 和 大拇指指尖下的关节3 的水平位置进行判断
            再根据 大拇指指尖4 和 食指指根5 的竖直距离进行判断
            综合上述两个条件才满足大拇指闭合！
            '''
            # 判断左右手，1为左手0为右手
            direction = 1 if hand[5][1] > hand[9][1] else 0
            for tip in tip_idx:
                # 获取每个指尖的位置并画圆
                x, y = hand[tip][1], hand[tip][2]
                cv.circle(img, (x, y), 20, (0, 255, 0), 2)

                # 判断每个指头是否打开
                # 特判大拇指，需要区分左右手
                if tip == 4:
                    if direction:
                        fingers.append(0 if hand[tip][1] < hand[tip - 1][1] and hand[tip][2] > hand[tip + 1][2] else 1)
                    else:
                        fingers.append(0 if hand[tip][1] > hand[tip - 1][1] and hand[tip][2] > hand[tip + 1][2] else 1)
                    continue
                # 其他手指
                fingers.append(1 if hand[tip][2] < hand[tip - 2][2] else 0)

            # 获取手势显示图片
            fingerImg = fingersImgLst[fingers.count(1)]
            # 获得图片的高宽
            h, w, _ = fingerImg.shape
            # 根据左右手判断图片出现位置，左手左上角右手右上角
            if direction:
                img[0:h, 0:w] = fingerImg
            else:
                screen_h, screen_w, _ = img.shape
                img[0:h, screen_w - w:] = fingerImg

        cv.imshow('Image', img)
        key = cv.waitKey(1)
        # 键盘输入q时退出
        if key == ord('q'):
            break
        # 键盘输入s时保存
        elif key == ord('s'):
            save_nums += 1
            cv.imwrite('./data/myHand_%d.jpg' % save_nums, img)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
