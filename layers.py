import tensorflow as tf
from keras_cv_attention_models import *
from keras import *
from keras.layers import *
import keras
import tensorflow_addons as tfa

from settings import *


class SelfAttention(Layer):
    def __init__(self, trainable=True):
        super().__init__(trainable=trainable)
        self.f = None
        self.g = None
        self.h = None
        self.v = None
        self.attention = None

    def build(self, input_shape):
        c = input_shape[-1]
        self.f = self.block(c // 8)  # reduce channel size, reduce computation
        self.g = self.block(c // 8)  # reduce channel size, reduce computation
        self.h = self.block(c // 8)  # reduce channel size, reduce computation
        self.v = Conv2D(c, 1, 1)  # scale back to original channel size

    @staticmethod
    def block(c):
        return keras.Sequential([
            Conv2D(c, 1, 1, padding="same"),  # [n, w, h, c] 1*1conv
            Reshape((-1, c)),  # [n, w*h, c]
        ])

    def call(self, inputs, **kwargs):
        f = self.f(inputs)  # [n, w, h, c] -> [n, w*h, c//8]
        g = self.g(inputs)  # [n, w, h, c] -> [n, w*h, c//8]
        h = self.h(inputs)  # [n, w, h, c] -> [n, w*h, c//8]
        s = tf.matmul(f, g, transpose_b=True)  # [n, w*h, c//8] @ [n, c//8, w*h] = [n, w*h, w*h]
        self.attention = tf.nn.softmax(s, axis=-1)
        context_wh = tf.matmul(self.attention, h)  # [n, w*h, w*h] @ [n, w*h, c//8] = [n, w*h, c//8]
        s = inputs.shape  # [n, w, h, c]
        cs = context_wh.shape  # [n, w*h, c//8]
        context = tf.reshape(context_wh, [-1, s[1], s[2], cs[-1]])  # [n, w, h, c//8]
        o = self.v(context) + inputs  # residual
        return o


def rot180_concat(inputs):
    x = tf.image.rot90(inputs, k=2)
    x = Concatenate()([x, inputs])
    return x


def rot180_add(inputs):
    x = tf.image.rot90(inputs, k=2)
    x = inputs + x
    return x


def vh_reduce(inputs):
    print(inputs.shape)
    _, h, w, d = inputs.shape
    x = DepthwiseConv2D(kernel_size=(1, h), padding="valid", activation="linear")(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(w, 1), padding="valid", activation="linear")(x)
    x = BatchNormalization()(x)
    return x


def soft_cce(y_true, y_pred, alpha=0.0, **kwargs):
    y_true = tf.cast(y_true, tf.float32)
    y_true = (1-alpha) * y_true + alpha * y_pred
    return tf.keras.losses.CategoricalCrossentropy(**kwargs)(y_true, y_pred)

