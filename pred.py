#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 02:36:13 2021

@author: nextgen
"""


import os

import configs

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)


import re

import glob

import tarfile

import os
import warnings
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random
from PIL import Image
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np

from os import listdir
from os.path import isfile, join


import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


def unet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = tf.keras.Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)


    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)
    final = UpSampling3D(size=(1,1,3))(conv10)

    model = Model(inputs = inputs, outputs = final)

    #model.summary()
    if(pretrained_weights):

    	model.load_weights(pretrained_weights)

    return model

def unet_cbam(pretrained_weights=None, input_size=(512, 512, 3), kernel_size=3, ratio=3, activ_regularization=0.01):
    inputs = tf.keras.Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    conv1 = CBAM_attention(conv1, ratio, kernel_size, dr_ratio, activ_regularization)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

    conv2 = CBAM_attention(conv2, ratio, kernel_size, dr_ratio, activ_regularization)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    conv3 = CBAM_attention(conv3, ratio, kernel_size, dr_ratio, activ_regularization)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    conv4 = CBAM_attention(conv4, ratio, kernel_size, dr_ratio, activ_regularization)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    conv6 = CBAM_attention(conv6, ratio, kernel_size, dr_ratio, activ_regularization)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    conv7 = CBAM_attention(conv7, ratio, kernel_size, dr_ratio, activ_regularization)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))

    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv8))

    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)

    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='relu')(conv9)

    final = UpSampling3D(size=(1, 1, 3))(conv10)

    model = Model(inputs=inputs, outputs=final)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

dr_ratio = 0.2
ratio=8
activ_regularization=0.0001
kernel_size=7
kernel_initializer = tf.keras.initializers.VarianceScaling()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def CBAM_attention(inputs,ratio,kernel_size,dr_ratio,activ_regularization):
    x = inputs
    channel = x.get_shape()[-1]

    ##channel attention##
    avg_pool = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    avg_pool = Dense(units = channel//ratio ,activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), use_bias=True,bias_initializer='zeros',trainable=True)(avg_pool)
    avg_pool = Dense(channel, activation = 'relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True, bias_initializer='zeros',trainable=True)(avg_pool)

    max_pool = tf.reduce_max(x, axis=[1,2], keepdims=True)
    max_pool = Dense(units = channel//ratio, activation='relu', activity_regularizer=tf.keras.regularizers.l1(activ_regularization),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), use_bias=True,bias_initializer='zeros',trainable=True)(max_pool)
    max_pool = Dense(channel, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), activity_regularizer=tf.keras.regularizers.l1(activ_regularization),use_bias=True, bias_initializer='zeros',trainable=True)(max_pool)
    f = Add()([avg_pool, max_pool])
    f = Activation('sigmoid')(f)

    after_channel_att = multiply([x, f])

    ##spatial attention##
    kernel_size = kernel_size
    avg_pool_2 = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    max_pool_2 = tf.reduce_max(x, axis=[1,2], keepdims=True)
    concat = tf.concat([avg_pool,max_pool],3)
    concat = Conv2D(filters=1, kernel_size=[kernel_size,kernel_size], strides=[1,1], padding='same', kernel_initializer=kernel_initializer,use_bias=False)(concat)
    concat = Activation('sigmoid')(concat)
    ##final_cbam##
    attention_feature = multiply([x,concat])
    return attention_feature


checkpoint_path1 = './pretrained_weights_nih'
checkpoint_dir1 = os.path.join(os.getcwd()+checkpoint_path1)
checkpoint_path2 = './pretrained_weights'
checkpoint_dir2 = os.path.join(os.getcwd()+checkpoint_path2)
cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path1,
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                mode='min')
cp_callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path2,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max')




print('------------------loading data--------------------')

#data
xr_train_rgb = np.load('xr_train.npy')
xr_valid_rgb = np.load('xr_valid.npy')
xr_test_rgb = np.load('xr_test.npy')
labels_train_xr = np.load('labels_train_xr.npy')
labels_valid_xr = np.load('labels_valid_xr.npy')
labels_test_xr = np.load('labels_test_xr.npy')

print('------------------building model--------------------')

'''
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_12.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.01 )

    unet_cbam_classification = Model(inputs = unet_cbam_model.input, outputs= unet_cbam_model.get_layer('dropout_1').output)

    my_model3 = tf.keras.Sequential()

    my_model3.add(unet_cbam_classification)

    my_model3.add(Conv2D(256,2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))

    #my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None, ratio=2))

    my_model3.add(Conv2D(128,2, activation = 'relu', padding = 'same',  kernel_initializer = 'he_normal'))

    #my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None,ratio=2))

    my_model3.add(MaxPooling2D(pool_size=(2,2)))

    my_model3.add(GlobalAveragePooling2D())

    my_model3.add(Dropout(0.3))

    my_model3.add(Dense(128,  activation = 'relu'))

    my_model3.add(Dropout(0.5))

    my_model3.add(Dense(64,  activation='relu'))

    #my_model3.add(Dropout(0.5))

    my_model3.add(Dense(32, activation='relu' ))

    #my_model3.add(Dropout(0.5))

    my_model3.add(Dense(16, activation='relu' ))

    my_model3.add(Dense(8, activation = 'relu' ))

    #my_model3.add(Dropout(0.3))

    my_model3.add(BatchNormalization())

    my_model3.add(Dense(3, activation = 'softmax'))

    my_model3.layers[0].trainable = True

    my_model3.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

'''

activ_regularization=0.00001

unet_cbam_model = unet_cbam(pretrained_weights = 'nih_unet_cbam_20.h5', input_size=(512,512,3),kernel_size=3, ratio=3, activ_regularization=0.000001 )

unet_cbam_classification = Model(inputs = unet_cbam_model.input, outputs= unet_cbam_model.get_layer('dropout_1').output)

my_model3 = tf.keras.Sequential()

my_model3.add(unet_cbam_classification)

my_model3.add(Conv2D(512,(2,2), activation = 'relu' ,padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

my_model3.add(Conv2D(256,(2,2), activation = 'relu' , padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None, ratio=2))

my_model3.add(BatchNormalization())

my_model3.add(Conv2D(128,(2,2), activation = 'relu' , padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

my_model3.add(Conv2D(64,(2,2), activation = 'relu' , padding = 'same',  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None,ratio=2))

my_model3.add(BatchNormalization())

my_model3.add(MaxPooling2D(pool_size=(2,2)))

my_model3.add(GlobalAveragePooling2D())

my_model3.add(BatchNormalization())

my_model3.add(Flatten())

my_model3.add(Dropout(0.3))

my_model3.add(Dense(128, activation = 'relu', activity_regularizer=tf.keras.regularizers.l2(activ_regularization),kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model3.add(Dropout(0.5))

my_model3.add(Dense(64,  activation='relu',activity_regularizer=tf.keras.regularizers.l2(activ_regularization),  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)))

#my_model3.add(Dropout(0.4))

my_model3.add(Dense(32, activation='relu' ,activity_regularizer=tf.keras.regularizers.l2(activ_regularization)))

my_model3.add(Dropout(0.3))

my_model3.add(Dense(16, activation='relu' ))

my_model3.add(Dense(8, activation = 'relu'))

#my_model3.add(Dropout(0.2))

#my_model3.add(BatchNormalization())

my_model3.add(Dense(3, activation='softmax'))

my_model3.layers[0].trainable = True

my_model3.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.000001, clipvalue=2, clipnorm=1),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


checkpoint_path13 = './training_1/cp7.ckpt'
my_model3.load_weights(checkpoint_path13)

sc = my_model3.predict(xr_test_rgb)
sc1 = my_model3.predict_classes(xr_test_rgb)

dfsc = pd.DataFrame(sc)
dfsc1 = pd.DataFrame(sc1)
dfl = pd.DataFrame(labels_test_xr)

dfsc.to_excel('logits_test.xlsx')
dfsc1.to_excel('pred_classes_test.xlsx')
dfl.to_excel('labaels_test.xlsx')

