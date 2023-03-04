import numpy as np
from keras import layers, Sequential
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras.layers import *


class ConvNext_Block(tf.keras.Model):
    """
    Implementing the ConvNeXt block for

    Args:
        dim: No of input channels
        drop_path: stotchastic depth rate
        layer_scale_init_value=1e-6

    Returns:
        A conv block
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(ConvNext_Block, self).__init__(**kwargs)

        self.depthwise_convolution = layers.Conv2D(dim, kernel_size=7, padding="same", groups=dim)
        self.layer_normalization = layers.LayerNormalization(epsilon=1e-6)
        self.pointwise_convolution_1 = layers.Dense(4 * dim)
        self.GELU = layers.Activation("gelu")
        self.pointwise_convolution_2 = layers.Dense(dim)
        self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        if drop_path > 0.0:
            self.drop_path = (tfa.layers.StochasticDepth(drop_path))
        else:
            self.drop_path = layers.Activation("linear")

    def call(self, inputs):
        x = inputs
        x = self.depthwise_convolution(x)
        x = self.layer_normalization(x)
        x = self.pointwise_convolution_1(x)
        x = self.GELU(x)
        x = self.pointwise_convolution_2(x)
        x = self.gamma * x

        return inputs + self.drop_path(x)


def convnext_block(inputs, in_features, drop_path=0.0, layer_scale_init_value=1e-6):
    block = Sequential([
        DepthwiseConv2D(kernel_size=7, padding="same"),
        LayerNormalization(epsilon=1e-6),
        Dense(4 * in_features),
        Activation("gelu"),
        Dense(in_features)
    ])

    gamma = tf.Variable(layer_scale_init_value * tf.ones((in_features,)))

    x = gamma * block(inputs)
    return tfa.layers.StochasticDepth(drop_path=drop_path)([inputs, x])


def patchify_stem(dims):
    stem = keras.Sequential([
        Conv2D(dims[0], kernel_size=4, strides=4),
        LayerNormalization(epsilon=1e-6)
    ])
    return stem


def spatial_downsampling(stem, dims, kernel_size, stride):
    ds_layers = [stem]
    for dim in dims[1:]:
        layer = keras.Sequential([
            LayerNormalization(epsilon=1e-6),
            Conv2D(dim, kernel_size, stride),
        ])
        ds_layers.append(layer)
    return ds_layers


def conv_next(inputs, dims, drop_path_rate, depths, layer_scale_init_value):
    stages = []
    dropout_rates = np.linspace(0.0, drop_path_rate, sum(depths))
    x = inputs
    cur = 0
    for i in range(len(dims)):
        for j in range(depths[i]):
            x = convnext_block(x, dims[i], dropout_rates[cur + j], layer_scale_init_value)
            cur += 1


        stage = keras.Sequential([
            ConvNext_Block(dim=dims[i], drop_path=dropout_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]

        )
        stages.append(stage)
        cur += depths[i]
    return stages

inputs = layers.Input((32,32,3))
x = inputs

stem=patchify_stem(dims)

downsampling=spatial_downsampling(stem,dims,kernel_size=2,stride=2)

stages=ConvNext_Stages(dims,drop_path_rate,depths,layer_scale_init_value)

for i in range(len(stages)):
    x = downsampling[i](x)
    x = stages[i](x)

x = layers.GlobalAvgPool2D()(x)
x = layers.LayerNormalization(epsilon=1e-6)(x)

outputs = layers.Dense(10)(x)

ConvNeXt_model=keras.Model(inputs, outputs)