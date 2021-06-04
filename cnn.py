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

# Toggles
# If DROPOUT = 0, dropout layers will be excluded
# Else, if DROPOUT = X, dropout layers will be included with a rate of 0.0X
# DROPOUT must be of type int and 0 >= DROPOUT >= 9
# DROPOUT <= 2 is recommended
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


def MusicGenre_CNN(weights=None, input_tensor=None):
    #(weights='msd', input_tensor=None)
    '''Instantiate the MusicTaggerCNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 256-dim features.
    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

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

    if weights is None:
        # Create model
        x = Dense(10, activation='softmax', name='output')(x)
        # x = Dense(1, activation='relu', name='max')(x)
        model = Model(melgram_input, x)

        return model
    else:
        # Load input
        x = Dense(50, activation='sigmoid', name='output')(x)
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)
        initial_model.load_weights('weights/music_tagger_cnn_weights_%s.h5' % K._BACKEND,
                                   by_name=True)

        # Eliminate last layer
        pop_layer(initial_model)

        # Add new Dense layer
        last = initial_model.get_layer('Flatten_1')
        preds = (Dense(10, activation='sigmoid', name='preds'))(last.output)
        model = Model(initial_model.input, preds)

        return model
