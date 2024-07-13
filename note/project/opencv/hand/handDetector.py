import cv2 as cv
import mediapipe as mp


class HandDetector:
    # 初始化
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_con=0.5, track_con=0.5):
        '''
        手势识别类初始化

        注意版本差异：
        低版本中参数为四个，分别是：
        :param mode: 是否为静态图片，默认False
        :param max_hands: 最多识别多少个手
        :param detection_confidence: 最小检测信度值，默认0.5
        :param track_confidence: 最小追踪信度值，默认0.5

        高版本中参数为五个，多了一个模型复杂度：
        static_image_mode
        max_num_hands
        model_complexity=新增！只有两个值0或1可选
        min_detection_confidence
        min_tracking_confidence
        '''
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.hands = mp.solutions.hands.Hands(mode, max_hands, complexity, detection_con, track_con)

    def find_hands(self, img):
        '''
        检测手势
        :param img: 图片
        :return: 绘画了手势的图片
        '''
        # 传入的是bgr，改成rgb
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # 检测手势
        self.results = self.hands.process(imgRGB)
        # 判断是否有手
        if self.results.multi_hand_landmarks:
            # 遍历每只手
            for handlms in self.results.multi_hand_landmarks:
                '''
                绘制手势
                参数1：image
                参数2：landmark_list 手势列表
                参数3：connections 连线（可忽略）
                参数4：关节样式（可忽略）
                参数5：连线样式（可忽略）
                '''
                mp.solutions.drawing_utils.\
                    draw_landmarks(img, handlms,
                                   mp.solutions.hands.HAND_CONNECTIONS,
                                   mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                   mp.solutions.drawing_styles.get_default_hand_connections_style())

        return img

    def find_positions(self, img):
        '''
        :param img: 图片
        :param hand_no: 手编号（默认第0只手）
        :return: 手势数据列表
        '''
        self.handslst = []
        if self.results.multi_hand_landmarks:
            '''
            注意hand.landmark的返回值
            因为最多允许检测两只手，因此返回的是一个三维数组
                第一个纬度：手
                第二维度：关节
                第三维度：
                    x：相对于屏幕左上角的水平距离
                    y：相对于屏幕左上角的竖直距离
                    z：相对于掌根的距离
            '''
            # landmarks得到的是关节点在图片中的比例位置，乘上图片的实际尺寸才是真实坐标
            hands = self.results.multi_hand_landmarks
            # 循环每只手
            for hand in hands:
                lmslist = []
                # 循环每个关节
                for idx, lm in enumerate(hand.landmark):
                    h, w, _ = img.shape
                    # 获取具体坐标
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmslist.append([idx, cx, cy])
                self.handslst.append(lmslist)

        return self.handslst

