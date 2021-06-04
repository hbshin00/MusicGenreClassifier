import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers import ReLU
from keras.utils.data_utils import get_file
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TOGGLES
# If DROPOUT = 0, dropout layers will be excluded
# Else, if DROPOUT = X, dropout layers will be included with a rate of 0.0X
# DROPOUT must be of type int and 0 >= DROPOUT >= 9
# In general, DROPOUT <= 2 is recommended, however best observed observance so far has been with no dropout
DROPOUT = 1
# If ARCHITECTURE = 1, the 2-layer architecture will be used
# If ARCHITECTURE = 2, the 5-layer architecture will be used
ARCHITECTURE = 1

def pop_layer(model):
    if not model.outputs:
        raise Exception('Pop failed')
        #'Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False


def MusicGenre_CNN(input_tensor=None):

    input_shape = (128, 647, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(shape=input_tensor)

    channel_axis = 1
    freq_axis = 1
    time_axis = 2

    # Input block
    x = BatchNormalization(axis=time_axis, trainable=False)(melgram_input)

    if ARCHITECTURE == 1:
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', trainable=False)(x)
        x = BatchNormalization(axis=channel_axis, trainable=False)(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 4), trainable=False)(x)
        if DROPOUT > 0:
            rate = DROPOUT / 10
            x = Dropout(rate)(x)

        # Conv block 2
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', trainable=False)(x)
        x = BatchNormalization(axis=channel_axis, trainable=False)(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 4), trainable=False)(x)
        if DROPOUT > 0:
            rate = DROPOUT / 10
            x = Dropout(rate)(x)

    elif ARCHITECTURE == 2:
        # Conv block 1
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', trainable=False)(x)
        x = BatchNormalization(axis=channel_axis, trainable=False)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 4), trainable=False)(x)

        # Conv block 2
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', trainable=False)(x)
        x = BatchNormalization(axis=channel_axis, trainable=False)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 4), trainable=False)(x)

        # Conv block 3
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', trainable=False)(x)
        x = BatchNormalization(axis=channel_axis, trainable=False)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 4), trainable=False)(x)

        # Conv block 4
        x = Conv2D(filters=192, kernel_size=(3, 3), padding='same', trainable=False)(x)
        x = BatchNormalization(axis=channel_axis, trainable=False)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3, 5), trainable=False)(x)

        # Conv block 5
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', trainable=False)(x)
        x = BatchNormalization(axis=channel_axis, trainable=False)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(5, 2), trainable=False)(x)

    # Output
    x = Flatten(name='Flatten_1')(x)

    # Create model
    x = Dense(10, activation='softmax', name='output')(x)
    # x = Dense(1, activation='relu', name='max')(x)
    model = Model(melgram_input, x)

    return model
