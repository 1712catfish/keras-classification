def bn(inputs):
    return BatchNormalization(epsilon=1.001e-5)(inputs)


def swish(inputs):
    return Activation("swish")(inputs)


def conv(filters, kernel_size, inputs):
    return Conv2D(filters, kernel_size, padding="same", **kwargs)(inputs)


def conv1x1(filters, inputs):
    return conv(filters, 1, inputs)


def max_pool_same_shape(pool_size, inputs):
    return MaxPooling2D(pool_size, strides=1, padding="same")(inputs)


def pool_pool_conv(filters, inputs):
    p2 = max_pool_same_shape(2, inputs)
    p4 = max_pool_same_shape(4, inputs)
    out = Concatenate()([inputs, p2, p4])
    out = conv1x1(filters, out)
    return out


def stack_fn(in_features, out_features, inputs):
    out = bn(inputs) + pool_pool_conv(in_features, inputs)
    out = bn(out) + conv1x1(out_features, inputs)
    out = swish(out)
    return out


def pool_net(inputs, dropout=0.4):
    out_channels = [16, 32, 48, 96, 112, 192]
    out = conv1x1(16, inputs)

    for i in range(len(out_channels) - 1):
        out = stack_fn(out_channels[i], out_channels[i + 1], out)
        out = Dropout(dropout)(out)
        out = stack_fn(out_channels[i + 1], out_channels[i + 1], out)
        if i < len(out_channels) - 2:
            out = MaxPooling2D()(out)

    return out
