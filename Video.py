import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from threading import Thread
from Prediction import *
import time
import dlib
import numpy as np
import pandas as pd
import os

class Video(QObject):
    sendImage = pyqtSignal(QImage)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    def __init__(self, widget, size):
        super().__init__()
        self.widget = widget
        self.size = size
        self.x = 0
        self.y = 0
        self.sendImage.connect(self.widget.recvImage)
        self.x_data = np.empty(4)
        
    def getInput(self):
        return self.x_data
    
    def startCam(self):
        try:
            self.cap = cv2.VideoCapture(0)
        except Exception as e:
            print('Cam error: ', e)
        else:
            self.bThread = True
            self.thread = Thread(target=self.threadFunc)
            self.thread.start()

    def stopCam(self):
        self.bThread = False
        boepn = False
        try:
            bopen = self.cap.isOpened()
        except Exception as e:
            print('Error cam not opened: ', e)
        else:
            self.cap.release()
            
    def get_rect(self,shape):
        rw = 0
        rh = 0
        rx = 65535
        ry = 65535
        for (x,y) in shape:
            rw = max(rw,x)
            rh = max(rh,y)
            rx = min(rx,x)
            ry = min(ry,y)
        return (rx,ry,rw-rx,rh-ry)
    
    def shape_to_np(self, shape, dtype="int"):
        #initialize the list of (x, y)-coordinates
        coords = np.zeros((20, 2), dtype=int)
        # for only lip landmarks
        n = 0
        for i in range(48, shape.num_parts):
            coords[n] = (shape.part(i).x, shape.part(i).y)
            n += 1
            
        # return the list of (x, y)-coordinates
        return coords
    
    def crop_LipImage(self, image, shape, language):
        img = image
        if language == "eng":
            (x, y, w, h) = self.get_rect(shape)
            img = image[y:y+h, x:x+w].copy()
            img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_AREA)
            
        if language == "kor":
            # select center of mouth
            x_list = [x[0] for x in shape]
            y_list = [y[1] for y in shape]
            self.x = sum(x_list)//20
            self.y = sum(y_list)//20
            img = image[self.y-50:self.y+50, self.x-100:self.x+100].copy()

        return img
    
    def threadFunc(self):
        isFirst = True
        while self.bThread:
            ok, frame = self.cap.read()
            if ok:
                image = cv2.resize(frame, dsize=(640, 480), interpolation = cv2.INTER_AREA)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_detector = self.detector(img_gray, 1)
                if len(face_detector) == 0:
                    print("***************No face Detected*************")
                
                for face in face_detector:
                    landmarks = self.predictor(image, face)
                    landmarks = self.shape_to_np(landmarks)
                    dst = self.crop_LipImage(image, landmarks, self.widget.language)
                    # create image
                    img_tensor = img_to_array(dst)
                    img_tensor /= 255.
                    img_tensor = np.expand_dims(img_tensor, axis = 0)

                    if isFirst:
                        self.x_data = img_tensor
                        isFirst = False
                    else:
                        self.x_data = np.concatenate((self.x_data, img_tensor), axis=0)
                
                # resize&send PyQt cam image
                h, w, ch = image.shape
                bytesPerLine = ch * w
                img = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
                resizedImg = img.scaled(w, h, Qt.KeepAspectRatio)
                self.sendImage.emit(img)
            else:
                print("cam read error")
            time.sleep(0.01) 
        print('thread finished')
