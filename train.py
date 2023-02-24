from keras.layers import *
from keras import *
from keras_cv_attention_models import *

from metrics import macro_f1
from settings import *


def create_model():
    inputs = Input((IMSIZE, IMSIZE, 3))
    #     x = rot180_concat(inputs)
    x = efficientnet.EfficientNetV2B2(
        input_shape=(IMSIZE, IMSIZE, 3),
        num_classes=0,
        pretrained="imagenet",
    )(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(NUM_CLASSES, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", macro_f1()]
    )

    return model


# model = create_model()
# model.summary()
