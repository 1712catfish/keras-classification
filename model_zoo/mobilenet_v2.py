def swish(inputs):
    return Activation('swish')(inputs)


def batch_norm(inputs):
    return BatchNormalization(renorm=True)(inputs)


def expansion_block(x, t, filters):
    x = Conv2D(filters * t, 1, padding='same', use_bias=False)(x)
    x = batch_norm(x)
    x = swish(x)
    return x


def depthwise_block(x, stride):
    x = DepthwiseConv2D(3, stride, padding='same', use_bias=False)(x)
    x = batch_norm(x)
    x = swish(x)
    return x


def projection_block(x, out_channels):
    x = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = batch_norm(x)
    return x


def bottleneck(x, t, out_channels, stride):
    filters = x.shape[-1]
    y = expansion_block(x, t, filters)
    y = depthwise_block(y, stride)
    y = projection_block(y, out_channels)
    if y.shape[-1] == x.shape[-1]:
        y = add([x, y])
    return y


def mobilenetv2(inputs):
    x = Conv2D(32, 3, 2, padding='same', use_bias=False)(inputs)
    x = batch_norm(x)
    x = swish(x)

    x = depthwise_block(x, 1)
    x = projection_block(x, 16)

    x = bottleneck(x, 6, 24, 2)
    x = bottleneck(x, 6, 24, 1)

    x = bottleneck(x, 6, 32, 2)
    x = bottleneck(x, 6, 32, 1)
    x = bottleneck(x, 6, 32, 1)

    x = bottleneck(x, 6, 64, 2)
    x = bottleneck(x, 6, 64, 1)
    x = bottleneck(x, 6, 64, 1)
    x = bottleneck(x, 6, 64, 1)

    x = bottleneck(x, 6, 96, 2)
    x = bottleneck(x, 6, 96, 1)
    x = bottleneck(x, 6, 96, 1)

    x = bottleneck(x, 6, 160, 2)
    x = bottleneck(x, 6, 160, 1)
    x = bottleneck(x, 6, 160, 1)

    x = bottleneck(x, 6, 320, 1)

    x = Conv2D(1280, 1, padding='same', use_bias=False)(x)
    x = batch_norm(x)
    x = swish(x)

    return x


with strategy.scope():
    inputs = Input((512, 512, 3))
    x = mobilenetv2(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()