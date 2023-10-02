# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import random
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.python.layers import utils
from tensorflow.keras import regularizers
from tensorflow.keras import layers

img_w = 384  
img_h = 384
n_label = 3


def channel_fc(data):
    out_dim = K.int_shape(data)
    squeeze = GlobalAveragePooling2D()(data)
    excitation = Dense(units=out_dim[3] // 4)(squeeze)
    # excitation = Dense(units=out_dim)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim[3])(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim[3]))(excitation)
    data_scale = multiply([data, excitation])
    return data_scale
    
def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def spatial_fc(data):
    out_dim = K.int_shape(data)
    squeeze = Conv2D(1, 1, strides=1, padding='same')(data)
    excitation = Activation('sigmoid')(squeeze)
    excitation = expend_as(excitation, out_dim[3])
    data_scale = multiply([data, excitation])
    return data_scale

def ConvBlock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data) #,dilation_rate=(4,4)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2

def updata(filte, data, skipdata):
    shape_x = K.int_shape(skipdata)
    shape_g = K.int_shape(data)
    up1 = UpSampling2D(size=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]))(data)
    concatenate = Concatenate()([up1, skipdata])
    concatenate = Conv2D(filte, (3, 3), padding="same")(concatenate)
    concatenate = BatchNormalization()(concatenate)
    concatenate = LeakyReLU(alpha=0.01)(concatenate)

    Selective_data = BAAF_fc_att(data=concatenate, size=filte)
    
    Selective_data = Concatenate()([Selective_data, up1])

    LeakyReLU2 = ConvBlock(Selective_data, filte)
    return LeakyReLU2

def BAAF_fc_att(data, size):
    out_dim = K.int_shape(data)
    r=8
    L=32 
    d = max(int(out_dim[3] / r), L)

    channel_data = channel_fc(data)
    spatial_data = spatial_fc(data)

    channel_data_x1 = GlobalAveragePooling2D()(channel_data)
    spatial_data_x1 = GlobalAveragePooling2D()(spatial_data)

    U = Add()([channel_data_x1, spatial_data_x1])

    z = Dense(d, activation='relu')(U)
    z = Dense(out_dim[3]*2)(z)

    z = Reshape([1, 1, out_dim[3], 2])(z)
    scale = Softmax()(z)

    x = Lambda(lambda x: tf.stack(x, axis=-1))([channel_data, spatial_data])

    r = multiply([scale, x])
    r = Lambda(lambda x: K.sum(x, axis=-1))(r)

    return r


def DeepBAAFNet_att():   
    inputs = Input((img_h, img_w, 3))
    Conv1 = ConvBlock(data=inputs, filte=64)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = ConvBlock(data=pool1, filte=128)

    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Conv3 = ConvBlock(data=pool2, filte=128)

    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)   
    Conv4 = ConvBlock(data=pool3, filte=256)

    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4)    
    Conv5 = ConvBlock(data=pool4, filte=256)

    pool5 = MaxPooling2D(pool_size=(2, 2))(Conv5)    
    Conv6 = ConvBlock(data=pool5, filte=512)

    pool6 = MaxPooling2D(pool_size=(2, 2))(Conv6)    
    Conv7 = ConvBlock(data=pool6, filte=512)

    pool7 = MaxPooling2D(pool_size=(2, 2))(Conv7)    
    Conv8 = ConvBlock(data=pool7, filte=1024)

    # 6
    up1 = updata(filte=512, data=Conv8, skipdata=Conv7)

    # 12
    up2 = updata(filte=512, data=up1, skipdata=Conv6)

    # 25
    up3 = updata(filte=256, data=up2, skipdata=Conv5)

    # 48
    up4 = updata(filte=256, data=up3, skipdata=Conv4)

    # 96
    up5 = updata(filte=128, data=up4, skipdata=Conv3)

    # 192
    up6 = updata(filte=128, data=up5, skipdata=Conv2)

    # 384
    up7 = updata(filte=64, data=up6, skipdata=Conv1)

    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up7)
    out = Activation('sigmoid')(outconv)

    model = Model(inputs=inputs, outputs=out)
    return model
