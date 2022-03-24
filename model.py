#!/usr/bin/env python

#####################################################################################
# MIT License
#
# Copyright (c) 2021 Nhat Truong Pham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# If you use this code or part of it, please cite the following paper:
# Thanh Tran, Nhat Truong Pham, and Jan Lundgren, "Detecting Drill Failure in the 
# Small Short-soundDrill Dataset", arXiv preprint arXiv:2108.11089, 2021.
#
#####################################################################################

from keras.layers import Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPooling2D, \
    Flatten, Activation, LeakyReLU, Reshape, LSTM
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

# =====================================================================================

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def CNN_LeakyReLU(params_learn=None, params_extract=None):
    """

    :param params_learn:
    :param params_extract:
    :return:
    """
    channel_axis = 1
    input_shape = (params_extract.get('patch_len'), params_extract.get('n_mels'), channel_axis)

    numClasses = params_learn.get('n_classes')

    spectro = Input(shape=input_shape)
    featureX = spectro

    # layer 1
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 2
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 3
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(256, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)

    # flattern
    featureX = Flatten()(featureX)

    # FC1
    featureX = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(featureX)

    featureX = Dropout(0.5)(featureX)

    # FC2
    out = Dense(numClasses,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(featureX)

    model = Model(inputs=spectro, outputs=out)

    return model


def CNN_LSTM_LeakyReLU(params_learn=None, params_extract=None):
    """

    :param params_learn:
    :param params_extract:
    :return:
    """
    channel_axis = 1
    input_shape = (params_extract.get('patch_len'), params_extract.get('n_mels'), channel_axis)

    numClasses = params_learn.get('n_classes')

    spectro = Input(shape=input_shape)
    featureX = spectro

    # layer 1
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 2
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 3
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(256, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)

    # CNN to LSTM
    featureX = Reshape(target_shape=(150, 256))(featureX)
    lstm = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm')(featureX)

    # flatten
    featureX = Flatten()(lstm)

    # FC1
    featureX = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(featureX)

    featureX = Dropout(0.5)(featureX)

    # FC2
    out = Dense(numClasses,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(featureX)

    model = Model(inputs=spectro, outputs=out)

    return model


def CNN_LSTM_Att_LeakyReLU(params_learn=None, params_extract=None):
    """

    :param params_learn:
    :param params_extract:
    :return:
    """
    channel_axis = 1
    input_shape = (params_extract.get('patch_len'), params_extract.get('n_mels'), channel_axis)

    numClasses = params_learn.get('n_classes')

    spectro = Input(shape=input_shape)
    featureX = spectro

    # layer 1
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 2
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 3
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)
    featureX = Conv2D(256, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = LeakyReLU()(featureX)

    # fit CNN to LSTM
    featureX = Reshape(target_shape=(150, 256))(featureX)
    lstm = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm')(featureX)

    # attention
    attention = Attention(150)(lstm)

    # FC1
    featureX = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(attention)

    featureX = Dropout(0.5)(featureX)

    #FC2
    out = Dense(numClasses,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(featureX)

    model = Model(inputs=spectro, outputs=out)

    return model


def CNN_LSTM_Att_ReLU(params_learn=None, params_extract=None):
    """

    :param params_learn:
    :param params_extract:
    :return:
    """
    channel_axis = 1
    input_shape = (params_extract.get('patch_len'), params_extract.get('n_mels'), channel_axis)

    numClasses = params_learn.get('n_classes')

    spectro = Input(shape=input_shape)
    featureX = spectro

    # layer 1
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = Activation('relu')(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = Activation('relu')(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 2
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = Activation('relu')(featureX)
    featureX = Conv2D(128, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = Activation('relu')(featureX)
    featureX = MaxPooling2D(pool_size=(2, 4), data_format="channels_last")(featureX)

    # layer 3
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = Activation('relu')(featureX)
    featureX = Conv2D(256, (3, 3),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_last')(featureX)
    featureX = BatchNormalization(axis=-1)(featureX)
    featureX = Activation('relu')(featureX)

    # fit CNN to LSTM
    featureX = Reshape(target_shape=(150, 256))(featureX)
    lstm = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm')(featureX)

    # attention
    attention = Attention(150)(lstm)

    # FC1
    featureX = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(attention)

    featureX = Dropout(0.5)(featureX)

    # FC2
    out = Dense(numClasses,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(featureX)

    model = Model(inputs=spectro, outputs=out)

    return model
