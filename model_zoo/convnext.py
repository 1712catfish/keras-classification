import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.initializers.initializers_v2 import TruncatedNormal
from keras.layers import *
import tensorflow_addons as tfa
from tensorflow.python.keras.models import Model


class GRN(Layer):
    def __init__(self, dim):
        super().__init__(self)
        self.gamma = tf.Variable(lambda: tf.zeros((1, 1, 1, dim)))
        self.beta = tf.Variable(lambda: tf.zeros((1, 1, 1, dim)))

    def call(self, x, *args, **kwargs):
        Gx = tf.norm(x, ord=2, axis=(1, 2), keepdims=True)
        Nx = Gx / (tf.reduce_mean(Gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def init_weights(m):
    for layer in m.layers:
        if isinstance(layer, (Conv2D, Dense)):
            layer.kernel_initializer = TruncatedNormal(mean=0., stddev=.02)
    return m


class ConvNextV2_Block(Layer):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.drop_path = drop_path

    def call(self, inputs, *args, **kwargs):
        f = Sequential([
            DepthwiseConv2D(kernel_size=7, strides=1, padding='same', use_bias=False),
            LayerNormalization(),
            Dense(4 * self.dim),
            Activation('gelu'),
            GRN(4 * self.dim),
            Dense(self.dim),
        ])
        return tfa.layers.StochasticDepth(drop_path=self.drop_path)([inputs, f(inputs)])


def meta_conv_next_v2(
        block=ConvNextV2_Block,
        depths=None,
        dims=None,
        drop_path_rate=0.,
        **kwargs
):
    if depths is None:
        depths = [3, 3, 9, 3]

    if dims is None:
        dims = [96, 192, 384, 768]

    stem = Sequential([
        Conv2D(dims[0], 4, 4),
        LayerNormalization()
    ])

    downsample_layers = [stem]
    for dim in dims[1:]:
        downsample_layer = Sequential([
            LayerNormalization(),
            Conv2D(dim, 2, 2),
        ])
        downsample_layers.append(downsample_layer)

    stages = []
    dp_rates = np.linspace(0, drop_path_rate, sum(depths))
    cur = 0
    for i, (dim, depth) in enumerate(zip(dims, depths)):
        stage = Sequential([
            block(dim, dp_rates[cur + j])
            for j in range(depth)
        ])
        cur += depth
        stages.append(stage)

    def res(inputs):
        x = inputs
        for downsample_layer, stage in zip(downsample_layers, stages):
            x = downsample_layer(x)
            x = stage(x)
        return x

    return res


def meta_convnextv2_atto(**kwargs):
    model = meta_conv_next_v2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def meta_convnextv2_femto(**kwargs):
    model = meta_conv_next_v2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def meta_convnext_pico(**kwargs):
    model = meta_conv_next_v2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def meta_convnextv2_nano(**kwargs):
    model = meta_conv_next_v2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def meta_convnextv2_tiny(**kwargs):
    model = meta_conv_next_v2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def meta_convnextv2_base(**kwargs):
    model = meta_conv_next_v2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def meta_convnextv2_large(**kwargs):
    model = meta_conv_next_v2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def meta_convnextv2_huge(**kwargs):
    model = meta_conv_next_v2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


def conv_next_v2(**kwargs):
    return meta_conv_next_v2(block=ConvNextV2_Block, **kwargs)


def convnextv2_atto(**kwargs):
    model = meta_convnextv2_atto(block=ConvNextV2_Block, **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = meta_convnextv2_femto(block=ConvNextV2_Block, **kwargs)
    return model


def convnext_pico(**kwargs):
    model = meta_convnext_pico(block=ConvNextV2_Block, **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = meta_convnextv2_nano(block=ConvNextV2_Block, **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = meta_convnextv2_tiny(block=ConvNextV2_Block, **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = meta_convnextv2_base(block=ConvNextV2_Block, **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = meta_convnextv2_large(block=ConvNextV2_Block, **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = meta_convnextv2_huge(block=ConvNextV2_Block, **kwargs)
    return model


def meta_create_conv_next_v2(
        model=None,
        image_size=512,
        num_classes=1000,
):
    if model is None:
        model = meta_convnextv2_base()

    inputs = Input((image_size, image_size, 3))
    x = model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = LayerNormalization()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return init_weights(Model(inputs, x))
