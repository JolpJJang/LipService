import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from threading import Thread

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers, activations
from tensorflow.keras.layers import Conv2D, TimeDistributed, BatchNormalization, MaxPooling2D, Flatten, Bidirectional, Dense,Dropout,Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from threading import Thread
from MobileNet_LSTM import *

class Prediction(QObject):
    
    sendResult = pyqtSignal(str)
    words_kor = ['아','안','암','어','오','우','이','임','애','와','외']
    words_eng = ['Stop navigation', 'Excuse me.', 'I am sorry.', 'Thank you.', 'Good bye.', 'I love this game.', 'Nice to meet you.', 'You are welcome.', 'How are you?', 'Have a good time.']
    
    def __init__(self, widget, language):
        super().__init__()
        self.widget = widget
        self.language = language
        self.sendResult.connect(self.widget.getResult)
        self.mobilenet = MobileNet_LSTM(num_classes = 11)
        
    def predict(self, data):
        try:
            self.bThread = True
        except Exception as e:
            print("Predict Error: ", e)
        else:
            self.bThread = True
            self.thread = Thread(target=self.threadFunc, args=(data))
            self.thread.start()
            
    def threadFunc(self, data):
        while self.bThread:
            result = ""
            if self.language == "eng":
                # 차후 영어용 가중치로 바꾸기
                self.mobilenet.load_weights('eng/MobileNet_lip')
                output = self.mobilenet.predict(data)
                result = self.words_eng[np.argmax(output)]
                print("result: ", result)
                self.sendResult.emit(result)
                self.bThread = False

            if self.language == "kor":
                # 차후 한국어용 가중치로 바꾸기
                self.mobilenet.load_weights('kor/MobileNet_weights')
                output = self.mobilenet.predict(data)
                result = self.words_kor[np.argmax(output)]       
                print("output: ", output, "result: ", result)
                self.sendResult.emit(result)
                self.bThread = False