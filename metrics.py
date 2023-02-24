import tensorflow_addons as tfa
import tensorflow as tf

def macro_f1():
    return tfa.metrics.F1Score(
        num_classes=NUM_CLASSES,
        average="macro"
    )


learning_rate = 1e-3
epochs = 20

with strategy.scope():
    f1_score = tfa.metrics.F1Score(
        num_classes=NUM_CLASSES,
        threshold=None,
    )

    log_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f'./"{backbone.name}--{BATCH_SIZE}--{learning_rate}--log'
    ),

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{backbone.name}--{BATCH_SIZE}--{learning_rate}.h5",
        monitor='val_acc',
        mode='auto',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
        min_delta=0.0001,
    )

    categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.2,
    )
