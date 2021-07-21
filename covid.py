
import tensorflow as tf

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
mypath = '/home/ubuntu/Desktop/data/pjh/images/images'
targetdir= '/home/ubuntu/Desktop/data/pjh/images/images'
fileExt = r'.png'
onlyfiles = [os.path.join(mypath, _) for _ in os.listdir(mypath) if _.endswith(fileExt)]
onlyfiles1 = onlyfiles[0:-1000]
onlyfiles2 = onlyfiles[-1001:-1]

####################################################################################
#####model genesis
####################################################################################

from skimage.transform import resize

try:  # SciPy >= 0.19

    from scipy.special import comb



except ImportError:

    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """



     The Bernstein polynomial of n, i as a function of t



    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """



       Given a set of control points, return the



       bezier curve defined by the control points.







       Control points should be a list of lists, or list of tuples



       such as [ [1,1],



                 [2,3],



                 [4,5], ..[Xn, Yn] ]



        nTimes is the number of time steps, defaults to 1000







        See http://processingjs.nihongoresources.com/bezierinfo/



    """

    nPoints = len(points)

    xPoints = np.array([p[0] for p in points])

    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)

    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping

    cnt = 3

    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])

        x = np.flip(x, axis=degree)

        y = np.flip(y, axis=degree)

        cnt = cnt - 1

    return x, y


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x

    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]

    xpoints = [p[0] for p in points]

    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=100000)

    if random.random() < 0.5:

        # Half change to get flip

        xvals = np.sort(xvals)



    else:

        xvals, yvals = np.sort(xvals), np.sort(yvals)

    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):

    if random.random() >= prob:

        return x

    image_temp = copy.deepcopy(x)

    orig_image = copy.deepcopy(x)

    img_rows, img_cols, img_deps = x.shape

    num_block = 500

    for _ in range(num_block):

        block_noise_size_x = random.randint(1, img_rows//10)

        block_noise_size_y = random.randint(1, img_cols//10)

        #block_noise_size_z = random.randint(0, 3)

        noise_x = random.randint(0, img_rows-block_noise_size_x)

        noise_y = random.randint(0, img_cols-block_noise_size_y)

        #noise_z = random.randint(0, img_deps-block_noise_size_z)

        if img_deps >3 :

            filters = 3

            window = orig_image[noise_x:noise_x+block_noise_size_x, 

                               noise_y:noise_y+block_noise_size_y, 

                               #noise_z:noise_z+block_noise_size_z,
                                :,
                           ]

            window = window.flatten()

            np.random.shuffle(window)

            window = window.reshape((block_noise_size_x, 

                                 block_noise_size_y, 
                                 img_deps))
                                 #block_noise_size_z))

            image_temp[noise_x:noise_x+block_noise_size_x, 

                      noise_y:noise_y+block_noise_size_y, 
                      :] = window
                      #noise_z:noise_z+block_noise_size_z] = window

        else :

            window = orig_image[noise_x:noise_x+block_noise_size_x, 

                               noise_y:noise_y+block_noise_size_y, 

                               #noise_z:noise_z+block_noise_size_z,
                                :,
                           ]

            window = window.flatten()

            np.random.shuffle(window)

            window = window.reshape((block_noise_size_x, 

                                 block_noise_size_y, 
                                 img_deps))

                                 #block_noise_size_z))

            image_temp[noise_x:noise_x+block_noise_size_x, 

                      noise_y:noise_y+block_noise_size_y, 
                       :] = window

                      #noise_z:noise_z+block_noise_size_z] = window

    local_shuffling_x = image_temp

 

    return local_shuffling_x


def image_in_painting(x):
    img_rows, img_cols, img_deps = x.shape

    cnt = 5

    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)

        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)

        block_noise_size_z = random.randint(0, 3)

        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)

        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)

        noise_z = random.randint(0, img_deps - block_noise_size_z)

        x[

        noise_x:noise_x + block_noise_size_x,

        noise_y:noise_y + block_noise_size_y,

        noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,

                                                               block_noise_size_y,

                                                               block_noise_size_z, ) * 1.0

        cnt -= 1

    return x


def image_out_painting(x):
    img_rows, img_cols, img_deps = x.shape

    image_temp = copy.deepcopy(x)

    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], ) * 1.0

    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)

    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)

    block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)

    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)

    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)

    noise_z = random.randint(0, img_deps - block_noise_size_z)

    x[

    noise_x:noise_x + block_noise_size_x,

    noise_y:noise_y + block_noise_size_y,

    noise_z:noise_z + block_noise_size_z] = image_temp[noise_x:noise_x + block_noise_size_x,

                                            noise_y:noise_y + block_noise_size_y,

                                            noise_z:noise_z + block_noise_size_z]

    cnt = 4

    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)

        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)

        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)

        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)

        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)

        noise_z = random.randint(0, block_noise_size_z)

        x[

        noise_x:noise_x + block_noise_size_x,

        noise_y:noise_y + block_noise_size_y,

        noise_z:noise_z + block_noise_size_z] = image_temp[noise_x:noise_x + block_noise_size_x,

                                                noise_y:noise_y + block_noise_size_y,

                                                noise_z:noise_z + block_noise_size_z]

        cnt -= 1

    return x


def generate_pair(img, batch_size, config, status="test"):
    img_rows, img_cols, img_deps = img.shape[1], img.shape[2], img.shape[3]

    while True:

        index = [i for i in range(img.shape[0])]

        random.shuffle(index)

        y = img[index[:batch_size]]

        x = copy.deepcopy(y)

        for n in range(config.batch_size):

            # Autoencoder

            x[n] = copy.deepcopy(y[n])

            # Flip

            x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel

            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)

            # Apply non-Linear transformation with an assigned probability

            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)

            # Inpainting & Outpainting

            if random.random() < config.paint_rate:

                if random.random() < config.inpaint_rate:

                    # Inpainting

                    x[n] = image_in_painting(x[n])



                else:

                    # Outpainting

                    x[n] = image_out_painting(x[n])

        '''

        # Save sample images module



        if config.save_samples is not None and status == "train" and random.random() < 0.01:



            n_sample = random.choice( [i for i in range(conf.batch_size)] )



            sample_1 = np.concatenate((x[n_sample,0,:,2*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,2*img_deps//6].reshape(x[n].shape[0],1)), axis=1)



            sample_2 = np.concatenate((x[n_sample,0,:,3*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,3*img_deps//6].reshape(x[n].shape[0],1)), axis=1)



            sample_3 = np.concatenate((x[n_sample,0,:,4*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,4*img_deps//6].reshape(x[n].shape[0],1)), axis=1)



            sample_4 = np.concatenate((x[n_sample,0,:,5*img_deps//6].reshape(x[n].shape[0],1), y[n_sample,0,:,5*img_deps//6].reshape(x[n].shape[0],1)), axis=1)



            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)



            #final_sample = final_sample * 255.0



            final_sample = final_sample.astype(np.float32)



            #file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples



            #imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)



         '''

        yield (x, y)


import os
import shutil


class models_genesis_config:
    model = "Vnet"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix

    # data
    data = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes"
    train_fold = [0, 1, 2, 3, 4]
    valid_fold = [5, 6]
    test_fold = [7, 8, 9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64
    input_deps = 32
    nb_class = 1

    # model pre-training
    verbose = 1
    weights = None
    batch_size = 6
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 1e1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4

    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    sample_path = "pair_samples"
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    shutil.rmtree(os.path.join(sample_path, exp_name), ignore_errors=True)
    if not os.path.exists(os.path.join(sample_path, exp_name)):
        os.makedirs(os.path.join(sample_path, exp_name))

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


##########################################################################

plt.imshow(xr_train_rgb[10])

#######unet


##########################################################################

import skimage.io as io

import skimage.transform as trans

from tensorflow.keras import Model

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.keras import backend as keras

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)

    except RuntimeError as e:

        print(e)


def unet(pretrained_weights=None, input_size=(512, 512, 3)):
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    #with mirrored_strategy.scope():
        inputs = tf.keras.Input(input_size)

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

        drop4 = Dropout(0.5)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))

        merge6 = concatenate([drop4, up6], axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))

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


def unet_cbam(pretrained_weights=None, input_size=(512, 512, 3), kernel_size=3, ratio=3, activ_regularization=0.01):
    inputs = tf.keras.Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(conv1)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv1)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv1, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##

    after_spatial_att = multiply([conv1, cbam_feature])

    pool1 = MaxPooling2D(pool_size=(2, 2))(after_spatial_att)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(conv2)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv2)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv2, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##

    after_spatial_att_2 = multiply([conv2, cbam_feature])

    pool2 = MaxPooling2D(pool_size=(2, 2))(after_spatial_att_2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(conv3)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv3)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv3, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##

    after_spatial_att_3 = multiply([conv3, cbam_feature])

    pool3 = MaxPooling2D(pool_size=(2, 2))(after_spatial_att_3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(conv4)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv4)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv4, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##

    after_spatial_att_4 = multiply([conv4, cbam_feature])

    drop4 = Dropout(0.5)(after_spatial_att_4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(conv5)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv5)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv5, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##

    after_spatial_att_5 = multiply([conv5, cbam_feature])

    drop5 = Dropout(0.5)(after_spatial_att_5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(conv6)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(conv6)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([conv6, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(0.2)(cbam_feature)

    ##final_cbam##

    after_spatial_att_6 = multiply([conv6, cbam_feature])

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(after_spatial_att_6))

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

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

os.chdir('/home/nextgen/Desktop/tf2/covid')
unet = unet(pretrained_weights= './unet_xr.h5' , input_size=(512, 512, 3))

unet = unet(pretrained_weights= './unet_xr.h5' , input_size=(512, 512, 3))

unet_cbam_model = unet_cbam(pretrained_weights='./pretrained_weights/nuet_xr.h5', input_size=(512, 512, 3),
                            kernel_size=3, ratio=3, activ_regularization=0.01)

unet_cbam_model = unet_cbam(pretrained_weights='./pretrained_weights/unet_ct_cbam.h5', input_size=(512, 512, 3),
                            kernel_size=3, ratio=3, activ_regularization=0.01)

for layer in unet.layers:
    print(layer.name)

unet_cbam_model.summary()

######################################################################################################################
###################################GradCAm############################################################################
######################################################################################################################\
unet.summary()
grad_model = tf.keras.models.Model([unet.inputs], [unet.get_layer('up_sampling3d').output, unet.output])

strategy = tf.distribute.MirroredStrategy()
a = grad_model(xr_train_rgb[0:1])
a[0].shape
import cv2
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array(xr_train_rgb[0:1]))
    loss = predictions[:,0]

output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

weights = tf.reduce_mean(guided_grads, axis=(0, 1))

cam = np.ones(output.shape[0: 2], dtype = np.float32)

for i, w in enumerate(weights):
    cam += w * output[:, :, i]
cam = cv2.resize(cam.numpy(), (512, 512))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)


cv2.imwrite('cam.png', output_image)



unet.compile(optimizer=Adam(lr=1e-4),

             loss='MSE',

             metrics=["MAE", "MSE"])

os.chdir('/home/nextgen/Desktop/tf2/covid')

import configs

models_genesis_config.nb_epoch // 500

checkpoint_path = 'training_1/cp.ckpt'

checkpoint_dir = os.path.join(os.getcwd() + checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,

                                                 save_weights_only=True,

                                                 monitor='val_loss',

                                                 save_best_only=True,

                                                 mode='min')

#from configs import models_genesis_config
while True:

    # To find a largest batch size that can be fit into GPU

    try:

        history = unet.fit(generate_pair(xr_train_rgb, batch_size=16,
                                         config=models_genesis_config, status="train"),

                           validation_data=generate_pair(xr_valid_rgb,
                                                         batch_size=models_genesis_config.batch_size,
                                                         config=models_genesis_config, status="test"),

                           validation_steps=xr_valid_rgb.shape[0] // models_genesis_config.batch_size,

                           steps_per_epoch=xr_train_rgb.shape[0] // models_genesis_config.batch_size,

                           epochs=12,

                           max_queue_size=models_genesis_config.max_queue_size,

                           workers=models_genesis_config.workers,

                           use_multiprocessing=True,

                           shuffle=True,

                           verbose=models_genesis_config.verbose

                           # callbacks=[cp_callback]

                           )

        break

    except tf.errors.ResourceExhaustedError as e:

        models_genesis_config.batch_size = int(models_genesis_config.batch_size - 1)

        print("\n> Batch size = {}".format(models_genesis_config.batch_size))


a = unet.predict(xr_test_rgb)

plt.imshow(xr_test_rgb[10])

plt.imshow(a[10])

for layer in unet.layers:
    print(layer.name)

unet_cbam_classification = Model(inputs=unet.input, outputs=unet.get_layer('dropout_3').output)
del unet

from tensorflow.keras.regularizers import l2

from tensorflow.keras.regularizers import l1

from tensorflow.keras.models import *

from tensorflow.keras.layers import *


def CBAM_attention(inputs, ratio, kernel_size, dr_ratio, activ_regularization):
    x = Input(shape=inputs.output.shape[1:])

    ##channel attention##

    avg_pool = GlobalAveragePooling2D()(x)

    avg_pool = Dense(avg_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(avg_pool.shape[-1], kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = GlobalMaxPooling2D()(x)

    max_pool = Dense(max_pool.shape[-1], activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(max_pool.shape[-1], activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('relu')(f)

    after_channel_att = multiply([x, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(after_channel_att)

    max_pool_2 = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(after_channel_att)

    concat = Concatenate(axis=3)([avg_pool_2, max_pool_2])

    cbam_feature = Conv2D(1, (kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu',
                          activity_regularizer=tf.keras.regularizers.l2(activ_regularization), use_bias=True,
                          kernel_initializer='he_normal', bias_initializer='zeros', trainable=True)(concat)

    cbam_feature = Dropout(dr_ratio)(cbam_feature)

    ##final_cbam##

    after_spatial_att = multiply([x, cbam_feature])

    return Model(inputs=x, outputs=after_spatial_att)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    gcm = tf.keras.Sequential()
    #unet_cbam_classification = Model(inputs=unet.input, outputs=unet.get_layer('dropout_5').output)
    gcm.add(tf.keras.layers.Input(shape=(512,512,3)))

    gcm.add(unet_cbam_classification)

    #gcm.add(CBAM_attention(gcm, ratio=2, kernel_size=2, dr_ratio=0, activ_regularization=0.000001))

    gcm.add(Dropout(0.5))

    gcm.add(BatchNormalization())

    gcm.add(Conv2D(256, (2, 2), activation='relu', activity_regularizer=l2(0.0001), padding='same',
               kernel_initializer='he_normal'))

    gcm.add(Dropout(0.5))

    gcm.add(BatchNormalization())

    # gcm.add(CBAM_attention(gcm,ratio=2,kernel_size=2,dr_ratio=0,activ_regularization=0))


    gcm.add(Conv2D(128, (2, 2), activation='relu', activity_regularizer=l2(0.0001), padding='same',
               kernel_initializer='he_normal'))

    # gcm.add(CBAM_attention(gcm,ratio=8,kernel_size=2,dr_ratio=0,activ_regularization=0))


    gcm.add(BatchNormalization())

    gcm.add(Conv2D(64, (2, 2), activation='relu', activity_regularizer=l2(0.0001), padding='same'))

    gcm.add(BatchNormalization())

    # gcm.add(CBAM_attention(gcm,ratio=8,kernel_size=2,dr_ratio=0,activ_regularization=0))


    gcm.add(Conv2D(64, (2, 2), activation='relu', activity_regularizer=l2(0.0001), padding='same'))

    gcm.add(BatchNormalization())

    # gcm.add(CBAM_attention(gcm,ratio=8,kernel_size=2,dr_ratio=0,activ_regularization=0))


    gcm.add(MaxPooling2D(pool_size=(2, 2)))

    gcm.add(Conv2D(32, (2, 2), activation='relu', activity_regularizer=l2(0.0001), padding='same'))

    gcm.add(BatchNormalization())

    gcm.add(Conv2D(16, (2, 2), activation='relu', padding='same'))

    gcm.add(BatchNormalization())

    gcm.add(GlobalAveragePooling2D())

    gcm.add(BatchNormalization())

    gcm.add(Dense(128, activation='relu'))

    gcm.add(Dense(64, activation='relu'))

    gcm.add(Dense(32, activation='relu'))

    gcm.add(BatchNormalization())

    gcm.add(Dense(8, activation='relu'))

    gcm.add(Dropout(0.5))

    gcm.add(Dense(3, activation='softmax'))

    gcm.layers[0].trainable = True

gcm.summary()
strategy = tf.distribute.MirroredStrategy()

optimizer = tf.keras.optimizers.Adam(0.0001)
gcm.compile(optimizer=optimizer,

            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

            metrics=['accuracy'])


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.



  Arguments:

      schedule: a function that takes an epoch index

          (integer, indexed from 0) and current learning rate

          as inputs and returns a new learning rate as output (float).

  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()

        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get the current learning rate from model's optimizer.

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # Call schedule function to get the scheduled learning rate.

        scheduled_lr = self.schedule(epoch, lr)

        # Set the value back to the optimizer before this epoch starts

        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)

        print("\nEpoch %05d: Learning rate is %6.8f." % (epoch, scheduled_lr))


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (50, 0.005),
    (60, 0.005),
    (70, 0.005),
    (80, 0.0044),
    (130, 0.0042),
    (150, 0.004),
    (200, 0.004),
    (250, 0.0038),
    (280, 0.0036),
    (300, 0.0035),
    (350, 0.0035),
    (400, 0.0025),
    (450, 0.0022),
    (700, 0.001)
]

my_model3 = tf.keras.Sequential()

my_model3.add(unet_cbam_classification)

my_model3.add(
    Conv2D(256, 2, activation='relu', padding='same', activity_regularizer=l2(0.0001), kernel_initializer='he_normal'))

# my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None, ratio=2))


my_model3.add(
    Conv2D(128, 2, activation='relu', padding='same', activity_regularizer=l2(0.0001), kernel_initializer='he_normal'))

# my_model2.add(cbam_tf.cbam_block(my_model2.output,name=None,ratio=2))


my_model3.add(MaxPooling2D(pool_size=(2, 2)))

my_model3.add(GlobalAveragePooling2D())

my_model3.add(Dropout(0.3))

my_model3.add(Dense(128, activity_regularizer=l2(0.0001), activation='relu'))

my_model3.add(Dropout(0.5))

my_model3.add(Dense(64, activity_regularizer=l2(0.001), activation='relu'))

# my_model3.add(Dropout(0.5))


my_model3.add(Dense(32, activation='relu'))

# my_model3.add(Dropout(0.5))


my_model3.add(Dense(8, activation='relu'))

my_model3.add(Dropout(0.3))

my_model3.add(BatchNormalization())

my_model3.add(Dense(3, activation='softmax', activity_regularizer=l2(0.001)))

my_model3.layers[0].trainable = True



optimizer = tf.keras.optimizers.Adam(0.005)
my_model3.compile(optimizer=optimizer,

            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

            metrics=['accuracy'])


my_history3 = my_model3.fit(

    # aug.flow(ct_train, np.array(ct_train_label), batch_size=16),

    xr_train_rgb , np.array(labels_train_xr), batch_size=16,

    validation_data=(xr_valid_rgb, np.array(labels_valid_xr)),

    steps_per_epoch=xr_train_rgb.shape[0] // 16,  # number of images comprising of one epoch

    validation_steps=xr_valid_rgb.shape[0] // 16,

    # callbacks=[cp_callback, CustomLearningRateScheduler(lr_schedule)],

    callbacks=[CustomLearningRateScheduler(lr_schedule)],

    epochs=500)

# fine tuning -> small lr prevents over fitting

# small lr -> stuck in local minimum

# 0.00001 -> 0.1 -> 0.00001

del vgg16_tune


strategy = tf.distribute.MirroredStrategy()

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (50, 0.01),
    (80, 0.005)
]


with strategy.scope():
    #inputs = Input(shape=(512,512,3))
    vgg16_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=(512,512,3)))
    vgg16_transfer = vgg16_model.output
    vgg16_transfer = AveragePooling2D(pool_size=(4, 4))(vgg16_transfer)
    vgg16_transfer = Flatten(name="flatten")(vgg16_transfer)
    #vgg16_transfer = Dense(128, activation="relu")(vgg16_transfer)
    #vgg16_transfer = BatchNormalization()(vgg16_transfer)
    #vgg16_transfer = Dense(64, activation="relu")(vgg16_transfer)
    #vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Dense(32, activation="relu")(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Dense(8, activation="relu")(vgg16_transfer)
    #vgg16_transfer = Dropout(0.2)(vgg16_transfer)
    vgg16_transfer = Dense(3, activation='softmax')(vgg16_transfer)
    vgg16_tune = Model(inputs = vgg16_model.input, outputs = vgg16_transfer)
    vgg16_tune.compile(optimizer=Adam(learning_rate=0.01),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

history = vgg16_tune.fit(x = xr_train_rgb, y = labels_train_xr, batch_size=8,
                            validation_data = (xr_valid_rgb, labels_valid_xr),
                            validation_steps = xr_valid_rgb.shape[0]//8,
                            steps_per_epoch = xr_train_rgb.shape[0]//8,
                            callbacks = [CustomLearningRateScheduler(lr_schedule)],
                            epochs=100)



with strategy.scope():
    inputs = Input(shape=(512,512,3))
    vgg16_transfer = Conv2D(64,  2,  strides = 2,  activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.01),
                            kernel_initializer='he_normal')(inputs)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Conv2D(32,  2, strides = 2,  activation='relu',
                            activity_regularizer=tf.keras.regularizers.l2(0.01),
                            kernel_initializer='he_normal')(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Conv2D(16,  2, strides = 2, activation='relu',
                            activity_regularizer=tf.keras.regularizers.l2(0.01),
                            kernel_initializer='he_normal')(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Conv2D(16, 2, strides=2, activation='relu',
                            activity_regularizer=tf.keras.regularizers.l2(0.01),
                            kernel_initializer='he_normal')(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    #vgg16_transfer = GlobalAveragePooling2D()(vgg16_transfer)
    vgg16_transfer = Flatten(name="flatten")(vgg16_transfer)
    vgg16_transfer = Dense(1024, activation="relu")(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Dense(512, activation="relu")(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Dense(256, activation="relu")(vgg16_transfer)
    vgg16_transfer = Dense(128, activation="relu")(vgg16_transfer)
    #vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Dense(64, activation="relu")(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Dense(32, activation="relu")(vgg16_transfer)
    vgg16_transfer = BatchNormalization()(vgg16_transfer)
    vgg16_transfer = Dense(8, activation="relu")(vgg16_transfer)
    #vgg16_transfer = Dropout(0.2)(vgg16_transfer)
    vgg16_transfer = Dense(3, activation='softmax')(vgg16_transfer)
    vgg16_tune = Model(inputs = inputs, outputs = vgg16_transfer)
    vgg16_tune.compile(optimizer=Adam(learning_rate=0.01),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
vgg16_tune.summary()

history = vgg16_tune.fit(x = xr_train_rgb, y = labels_train_xr, batch_size=8,
                            validation_data = (xr_valid_rgb, labels_valid_xr),
                            validation_steps = xr_valid_rgb.shape[0]//8,
                            steps_per_epoch = xr_train_rgb.shape[0]//8,
                            callbacks = [CustomLearningRateScheduler(lr_schedule)],
                            epochs=100)



test_preds = np.array([2, 2, 1, 0, 1, 1, 1, 0, 1, 0, 1, 2, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 2, 0, 0, 1, 0, 1, 2, 1, 0, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 0, 0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 1, 1, 2, 0, 2, 2, 2, 0, 0, 2, 1, 1, 0, 1, 1, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 1, 2, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 0, 1, 0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 1, 2, 0, 1])
test_label = np.array([2, 2, 1, 2, 1, 1, 1, 0, 1, 0, 1, 2, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 2, 0, 0, 1, 2, 1, 2, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 1, 2, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 2, 1, 0, 0, 0, 2, 2, 1, 0, 0, 1, 0, 0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 1, 0, 1, 1, 2, 0, 2, 1, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 0, 0, 1, 0, 1, 1, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 2, 2, 0, 0, 0, 2, 0, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 1, 2, 0, 1])

from sklearn.metrics import confusion_matrix, accuracy_score, auc, f1_score, roc_curve

accuracy_score(test_preds, test_label)
confusion_matrix(test_preds, test_label)

fpr, tpr, thresholds = roc_curve(test_label, test_preds, pos_label=2)
auc(fpr,tpr)
f1_score(test_preds, test_label, average='micro')

105/108

122/128

'''
97%
[124, 1,  0]
[ 2, 122, 3]
[ 1, 5, 105]
'''
total = 124+1+2+122+3+1+5+105
accuracy =  (124+122+105)/total
precision = 124/125
recall = 124/127)
f1score = 2*(precision*recall)/(precision+recall)
tp = 124
fp = 1
fn=3
tn = 235
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
tpr = tp/(tp+fn)
fpr = fp/(fp+tn)
plt.plot(tpr,fpr)


test_pred97 = np.array([2, 2, 1, 2, 1, 1, 1, 1,  1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 0, 2, 1, 1, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 0, 2, 2, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 0, 0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 1, 0, 1, 1, 2, 0, 2, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 1, 2, 1, 2, 2, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 1, 2, 0, 1])

test_label97 = np.array([2, 2, 0, 2, 1, 1, 1, 1, 1, 0, 1, 2, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 2, 0, 0, 1, 0, 1, 2, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 1, 2, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 0, 0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 1, 0, 1, 1, 2, 0, 2, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 1, 2, 1, 2, 2, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 1, 2, 0, 1])

accuracy_score(test_pred97, test_label97)

from sklearn.metrics import roc_auc_score, RocCurveDisplay
n_classes = 3
n_samples = len(test_label97)

fpr, tpr, thresholds = roc_curve(test_label97, test_pred97, pos_label=2)
auc(fpr,tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_label97))[:, i], np.array(pd.get_dummies(test_pred97))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc


from sklearn.metrics import classification_report

target_names = ['COVID-19', 'Pneumonia', 'Normal']
print(classification_report(test_label97,test_pred97, target_names=target_names,digits=4))


test_pred94 = np.array([2, 2, 1, 2, 1, 1, 1, 0, 1, 0, 1, 2, 2, 0, 0, 1, 2, 1, 2, 0, 2, 2, 2, 2, 0, 0, 1, 2, 1, 2, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 1, 2, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0, 1, 0, 0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 1, 0, 1, 1, 2, 0, 2, 1, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 1, 2, 2, 2, 2, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 0, 1, 0, 0, 2, 2, 1, 2, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 1, 2, 0, 1])

test_label94 = np.array([2, 2, 1, 2, 1, 1, 1, 0, 1, 0, 1, 2, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 2, 0, 0, 1, 2, 1, 2, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 1, 2, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 2, 1, 0, 0, 0, 2, 2, 1, 0, 0, 1, 0, 0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 2, 2, 1, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 0, 0, 0, 2, 1, 0, 1, 1, 2, 0, 2, 1, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 0, 0, 1, 0, 1, 1, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 2, 2, 0, 0, 0, 2, 0, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 0, 2, 2, 1, 2, 2, 0, 2, 0, 0, 1, 2, 0, 1])

accuracy_score(test_pred94, test_label94)

confusion_matrix(test_pred94, test_label94)
'''
[124,   1,   3],
[  1, 124,  12],
[  0,   2,  96]]
'''
tp = 96
fp = 2
fn= 15
tn = 250
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_label94))[:, i], np.array(pd.get_dummies(test_pred94))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc

y_score = test_pred94
y_test = test_label94

print(classification_report(test_label94,test_pred94, target_names=target_names,digits=4))

from scipy import interp
from itertools import cycle
n_classes=3

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_label94))[:, i], np.array(pd.get_dummies(test_pred94))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(np.array(pd.get_dummies(test_label94))[:, i], np.array(pd.get_dummies(test_pred94))[:, i])
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 3

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()





n_classes = 3

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(test_label94))[:, i].ravel(), np.array(pd.get_dummies(test_pred94))[:, i].ravel())
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(np.array(pd.get_dummies(test_label94))[:, i].ravel(), np.array(pd.get_dummies(test_pred94))[:, i].ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
1/125
363*0.97

def CBAM_attention(inputs, ratio, kernel_size, dr_ratio, activ_regularization):
    x = inputs

    channel = x.get_shape()[-1]

    ##channel attention##

    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    avg_pool = Dense(units=channel // ratio, activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(avg_pool)

    avg_pool = Dense(channel, kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(avg_pool)

    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)

    max_pool = Dense(units=channel // ratio, activation='relu',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization),
                     kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', trainable=True)(max_pool)

    max_pool = Dense(channel, activation='relu', kernel_initializer='he_normal',
                     activity_regularizer=tf.keras.regularizers.l1(activ_regularization), use_bias=True,
                     bias_initializer='zeros', trainable=True)(max_pool)

    f = Add()([avg_pool, max_pool])

    f = Activation('sigmoid')(f)

    after_channel_att = multiply([x, f])

    ##spatial attention##

    kernel_size = kernel_size

    avg_pool_2 = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    max_pool_2 = tf.reduce_max(x, axis=[1, 2], keepdims=True)

    concat = tf.concat([avg_pool, max_pool], 3)

    concat = Conv2D(filters=1, kernel_size=[kernel_size, kernel_size], strides=[1, 1], padding='same',
                    kernel_initializer=kernel_initializer, use_bias=False)(concat)

    concat = Activation('sigmoid')(concat)

    ##final_cbam##

    attention_feature = multiply([x, concat])

    return attention_feature





conf = np.array([[125/127,   2/127,   0],
       [  0, 121/123,   2/123],
       [  0,   4/113, 109/113]])
conf = conf
df_conf = pd.DataFrame(conf, index = ['COVID19', 'Pneumonia', 'Normal'], columns = ['COVID19', 'Pneumonia', 'Normal'])
cmap = plt.get_cmap('Blues')
plt.imshow(df_conf, interpolation='nearest', cmap=cmap)
plt.matshow(df_conf, cmap=cmap)
plt.show()


import seaborn as sn
sn.set(font_scale=1)
sn.heatmap(df_conf, annot=True, annot_kws={'size':12})
plt.show()

a = plot_confusion_matrix(conf)

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()