import cv2 as cv
import os
import numpy as np


def readIMage(path):
    faceSamples, ids = [], []
    imagePaths = []
    for f in os.listdir(path):
        fullpath = path + f
        imagePaths.append(fullpath)

    face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

    for imagePath in imagePaths:
        id = os.path.split(imagePath)[-1].split('.')[0].split('_')[0]
        img = cv.imread(imagePath)
        # 预处理
        # resized_img = cv.resize(img, (92, 112))
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faceSamples, np.array(ids))
    recognizer.write('./data/trainer_cat.yml')
