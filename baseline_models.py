#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:38:00 2021

@author: nextgen
"""
import tensorflow as tf
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


from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=8, verbose=1, mode='max', min_lr=5e-5)


from tensorflow.keras import layers, models, Model, optimizers
learning_rate= 5e-5
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

##################################################################
#####################Load data####################################
##################################################################
xr_train_rgb = np.load('/home/nextgen/Desktop/tf2/covid/xr_train.npy')
xr_valid_rgb = np.load('/home/nextgen/Desktop/tf2/covid/xr_valid.npy')
xr_test_rgb = np.load('/home/nextgen/Desktop/tf2/covid/xr_test.npy')
labels_train_xr = np.load('/home/nextgen/Desktop/tf2/covid/labels_train_xr.npy')
labels_valid_xr = np.load('/home/nextgen/Desktop/tf2/covid/labels_valid_xr.npy')
labels_test_xr = np.load('/home/nextgen/Desktop/tf2/covid/labels_test_xr.npy')

from tensorflow.keras.applications import VGG16
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Freeze four convolution blocks
for layer in vgg_model.layers[:6]:
    layer.trainable = False
# Make sure you have frozen the correct layers
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)
    
x = vgg_model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = Dense(256, activation='relu')(x)
x = Dense(3, activation='softmax')(x) # Softmax for multiclass
transfer_model = Model(inputs=vgg_model.input, outputs=x)

transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
history = transfer_model.fit(xr_train_rgb, labels_train_xr, batch_size = 16, epochs=50, validation_data=(xr_valid_rgb,labels_valid_xr))#, callbacks=[lr_reduce,checkpoint])

result = transfer_model.evaluate(xr_test_rgb,labels_test_xr, batch_size = 32)
print(result)

transfer_model.save_weights('lks_model.h5')
###0.9752

###metrics
transfer_model.load_weights('lks_model.h5')

test_pred = transfer_model.predict(xr_test_rgb)
test_pred = np.argmax(test_pred, axis=1)
test_labels = np.argmax(labels_test_xr, axis=1)

confusion_matrix(test_labels, test_pred)
'''
array([[124,   1,   0],
       [  1, 121,   5],
       [  0,   2, 109]])
'''
total = 124+1+1+121+5+2+109
accuracy = (109+247)/total
tp = 124
fp = 1
fn=1
tn = 237
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2*(precision*recall)/(precision+recall)
sensitivity = tp/(tp+fp)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

fpr, tpr, thresholds = roc_curve(test_labels,test_pred, pos_label=2)

auc(fpr, tpr)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_labels))[:, i], np.array(pd.get_dummies(test_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc


from tensorflow.keras.applications import MobileNet

mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
x = mobilenet_model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = Dense(256, activation='relu')(x)
x = Dense(3, activation='softmax')(x) # Softmax for multiclass
transfer_model = Model(inputs=mobilenet_model.input, outputs=x)
transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

history1 = transfer_model.fit(xr_train_rgb, labels_train_xr, batch_size = 16, epochs=50, validation_data=(xr_valid_rgb,labels_valid_xr))

result_mobilenet = transfer_model.evaluate(xr_test_rgb,labels_test_xr, batch_size = 32)
print(result_mobilenet)
#0.975

transfer_model.save_weights('mobilenet.h5')

###metrics
transfer_model.load_weights('mobilenet.h5')

test_pred = transfer_model.predict(xr_test_rgb)
test_pred = np.argmax(test_pred, axis=1)
test_labels = np.argmax(labels_test_xr, axis=1)

confusion_matrix(test_labels, test_pred)
'''
array([[125,   0,   0],
       [  0, 121,   6],
       [  2,   1, 108]])
'''
total = 125+6+121+2+1+109
accuracy = (109+248)/total
tp = 125
fp = 0
fn= 2
tn = 236
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2*(precision*recall)/(precision+recall)
f1score
sensitivity = tp/(tp+fp)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_labels))[:, i], np.array(pd.get_dummies(test_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc



from tensorflow.keras.applications import MobileNetV2

mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
x = mobilenet_model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = Dense(256, activation='relu')(x)
x = Dense(3, activation='softmax')(x) # Softmax for multiclass
transfer_model = Model(inputs=mobilenet_model.input, outputs=x)
transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

history1 = transfer_model.fit(xr_train_rgb, labels_train_xr, batch_size = 16, epochs=50, validation_data=(xr_valid_rgb,labels_valid_xr), callbacks=[lr_reduce])

result_mobilenetv2 = transfer_model.evaluate(xr_test_rgb,labels_test_xr, batch_size = 32)
print(result_mobilenetv2)

#0.9614

transfer_model.save_weights('mobilenetv2.h5')

###metrics
transfer_model.load_weights('mobilenetv2.h5')

test_pred = transfer_model.predict(xr_test_rgb)
test_pred = np.argmax(test_pred, axis=1)
test_labels = np.argmax(labels_test_xr, axis=1)

confusion_matrix(test_labels, test_pred)
'''
array([[125,   0,   0],
       [  1, 123,   3],
       [  1,  16,  94]])
'''

total = 125+1+123+3+1++16+94
accuracy = (109+248)/total
tp = 125
fp = 0
fn= 2
tn = 236
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2*(precision*recall)/(precision+recall)
f1score
sensitivity = tp/(tp+fp)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_labels))[:, i], np.array(pd.get_dummies(test_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc



from tensorflow.keras.applications import ResNet50
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    x = resnet_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=resnet_model.input, outputs=x)
    transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

history2 = transfer_model.fit(xr_train_rgb, labels_train_xr, batch_size = 4, epochs=30, validation_data=(xr_valid_rgb,labels_valid_xr))

result_resnet50 = transfer_model.evaluate(xr_test_rgb,labels_test_xr, batch_size = 4)
print(result_resnet50)
#0.9752

transfer_model.save_weights('resnet50.h5')

###metrics
transfer_model.load_weights('resnet50.h5')

test_pred = transfer_model.predict(xr_test_rgb)
test_pred = np.argmax(test_pred, axis=1)
test_labels = np.argmax(labels_test_xr, axis=1)

confusion_matrix(test_labels, test_pred)
'''
array([[125,   0,   0],
       [  1, 121,   5],
       [  1,   2, 108]])
'''

total = 125+1+121+5+1+2+108
accuracy = (109+248)/total
tp = 125
fp = 0
fn= 2
tn = 236
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2*(precision*recall)/(precision+recall)
f1score
sensitivity = tp/(tp+fp)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_labels))[:, i], np.array(pd.get_dummies(test_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc



from tensorflow.keras.applications import ResNet101
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    resnet_model = ResNet101(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    x = resnet_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=resnet_model.input, outputs=x)
    transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

history2 = transfer_model.fit(xr_train_rgb, labels_train_xr, batch_size = 4, epochs=50, validation_data=(xr_valid_rgb,labels_valid_xr))

result_resnet101 = transfer_model.evaluate(xr_test_rgb,labels_test_xr, batch_size = 4)
print(result_resnet101)
#0.

transfer_model.save_weights('resnet101.h5')

###metrics
transfer_model.load_weights('resnet101.h5')

test_pred = transfer_model.predict(xr_test_rgb)
test_pred = np.argmax(test_pred, axis=1)
test_labels = np.argmax(labels_test_xr, axis=1)

confusion_matrix(test_labels, test_pred)
accuracy_score(test_labels,test_pred)
'''
array([[124,   1,   0],
       [  0, 121,   6],
       [  1,   3, 107]])
'''

total = 125+1+121+5+1+2+108
accuracy = (109+248)/total
tp = 124
fp = 1
fn= 4
tn = 237
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2*(precision*recall)/(precision+recall)
f1score
sensitivity = tp/(tp+fp)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_labels))[:, i], np.array(pd.get_dummies(test_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc


from tensorflow.keras.applications import EfficientNetB0
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    resnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    x = resnet_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=resnet_model.input, outputs=x)
    transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])

history2 = transfer_model.fit(xr_train_rgb, labels_train_xr, batch_size = 4, epochs=30, validation_data=(xr_valid_rgb,labels_valid_xr))

result_resnet101 = transfer_model.evaluate(xr_test_rgb,labels_test_xr, batch_size = 4)
print(result_resnet101)
#0.

transfer_model.save_weights('resnet101.h5')

###metrics
transfer_model.load_weights('resnet101.h5')

test_pred = transfer_model.predict(xr_test_rgb)
test_pred = np.argmax(test_pred, axis=1)
test_labels = np.argmax(labels_test_xr, axis=1)

confusion_matrix(test_labels, test_pred)
accuracy_score(test_labels,test_pred)
'''
array([[124,   1,   0],
       [  0, 121,   6],
       [  1,   3, 107]])
'''

total = 125+1+121+5+1+2+108
accuracy = (110+248)/total
tp = 124
fp = 1
fn=1
tn = 237
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2*(precision*recall)/(precision+recall)
sensitivity = tp/(tp+fp)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_labels))[:, i], np.array(pd.get_dummies(test_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc
