import keras
from keras import models, layers, activations
import tensorflow as tf
import numpy as np
from keras.applications import VGG16



class CNN_LSTM(keras.Model):
    def __init__(self, input_shape, num_classes):
        super(CNN_LSTM, self).__init__()
        self.BZ = input_shape[0]
        self.frame = input_shape[1]

        self.conv1 = layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape)
        self.conv2 = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')
        self.conv3 = layers.Conv2D(128, kernel_size=(5, 5), activation='relu')
        self.lstm = layers.LSTM(256, return_sequences=False)
        # self.lstm2 = layers.LSTM(128, return_sequences = True)
        # self.lstm3= layers.LSTM(256, return_sequences = False)
        self.dense = layers.Dense(16, activation='softmax')

        # self.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metircs = ['accuracy'])

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.reshape(x, (self.BZ, self.frame, -1))
        x = self.lstm(x)
        x = self.dense(x)
        return x


class deep_CNN_LSTM(keras.Model):
    def __init__(self, input_shape, num_classes):
        super(deep_CNN_LSTM, self).__init__()
        self.BZ = input_shape[0]
        self.frame = input_shape[1]

        self.conv1 = layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape)
        self.conv2 = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')
        self.conv3 = layers.Conv2D(128, kernel_size=(5, 5), activation='relu')

        self.bilstm = layers.Bidirectional(layers.LSTM(256), merge_mode='concat')
        self.dense = layers.Dense(num_classes, activation='softmax')
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = layers.Dropout(0.2)

    def call(self, input):
        fm = []
        for x in input:
            x = self.max_pool(self.dropout(layers.BatchNormalization()(self.conv1(x))))
            x = self.max_pool(self.dropout(layers.BatchNormalization()(self.conv2(x))))
            x = self.max_pool(self.dropout(layers.BatchNormalization()(self.conv3(x))))
            fm.append(x)
        x = tf.reshape(fm, (self.BZ, self.frame, -1))

        x = self.bilstm(x)
        x = self.dense(x)
        return x


class VGG_LSTM(keras.Model):
    def __init__(self, input_shape, num_classes):
        super(VGG_LSTM, self).__init__()
        self.BZ = input_shape[0]
        self.frame = input_shape[1]

        self.vgg = VGG16(weights='imagenet', include_top=False,
                         input_shape=(input_shape[2], input_shape[3], input_shape[4]))
        self.vgg.trainable = False

        self.bilstm = layers.Bidirectional(layers.LSTM(256), merge_mode='concat')
        self.dense = layers.Dense(num_classes, activation='softmax')
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = layers.Dropout(0.2)

    def call(self, input):
        print(input.shape)
        fm = []
        for x in input:
            x = self.vgg(x)
            # print(x.shape)
            fm.append(x)
        x = tf.reshape(fm, (self.BZ, self.frame, -1))  # (32, 20, 2048)
        x = self.bilstm(x)
        x = self.dense(x)
        return x


class VGG_LSTM_fine(keras.Model):
    def __init__(self, input_shape, num_classes):
        super(VGG_LSTM, self).__init__()
        self.BZ = input_shape[0]
        self.frame = input_shape[1]

        self.vgg = VGG16(weights='imagenet', include_top=False,
                         input_shape=(input_shape[2], input_shape[3], input_shape[4]))
        self.vgg.trainable = False

        set_trainable = False
        for layer in self.vgg.layers:
            if layer.name == 'block5_conv1' or layer.name == 'block5_conv2' or layer.name == 'block5_conv3':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        # self.lstm = layers.LSTM(256, return_sequences = False)
        self.bilstm = layers.Bidirectional(layers.LSTM(256), merge_mode='concat')
        # self.lstm2 = layers.LSTM(128, return_sequences = True)
        # self.lstm3= layers.LSTM(256, return_sequences = False)
        self.dense = layers.Dense(num_classes, activation='softmax')
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = layers.Dropout(0.2)
        # self.flatten = layers.Flatten()

    def call(self, input):
        print(input.shape)
        fm = []
        for x in input:
            x = self.vgg(x)
            # print(x.shape)
            fm.append(x)
        x = tf.reshape(fm, (self.BZ, self.frame, -1))  # (32, 20, 2048)
        x = self.bilstm(x)
        x = self.dense(x)
        label = []
        for i in range(len(x)):
            label.append(np.argmax(x[i]))
        return label