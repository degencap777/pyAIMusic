from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
import tensorflow as tf


def lstm():
    """
    Basic LSTM model.
    :return: compiled LST model
    """
    model = Sequential()
    # config = tf.ConfigProto()
    # jit_level = tf.OptimizerOptions.ON_1
    # config.graph_options.optimizer_options.global_jit_level = jit_level
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model