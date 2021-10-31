#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dlib
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers, activations
from tensorflow.keras.layers import Conv2D, TimeDistributed, BatchNormalization, MaxPooling2D, Flatten, Bidirectional, Dense,Dropout,Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Input

class MobileNet_LSTM(keras.Model):
    def __init__(self, num_classes):
        super(MobileNet_LSTM, self).__init__()
        
        self.mobile = TimeDistributed(MobileNet(include_top=False))

        #self.mobile = TimeDistributed(MobileNet(#weights='imagenet', include_top=False))

        self.bilstm = layers.Bidirectional(layers.LSTM(256), merge_mode='concat')
        self.dense = layers.Dense(num_classes, activation='softmax')
        self.max = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))
        self.dropout = layers.Dropout(0.2)
        self.flat= TimeDistributed(Flatten())

    def call(self, x):
        x = self.mobile(x)
        x = self.max(x)
        x = self.dropout(x)      
        x = self.flat(x)
        x = self.dropout(x)
        x = self.bilstm(x)
        return self.dense(x)

