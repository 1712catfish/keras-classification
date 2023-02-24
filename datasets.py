import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


def count_data_items(filenames):
    return np.sum([int(x[:-6].split('-')[-1]) for x in filenames])


def decode_image(image_data):
    image = tf.image.decode_image(image_data, channels=3)
    image = tf.reshape(image, [IMSIZE, IMSIZE, 3])
    image = tf.cast(image, tf.float32) / 255.
    return image


FEATURE_MAP = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'complex': tf.io.FixedLenFeature([], tf.int64),
    'frog_eye_leaf_spot': tf.io.FixedLenFeature([], tf.int64),
    'powdery_mildew': tf.io.FixedLenFeature([], tf.int64),
    'rust': tf.io.FixedLenFeature([], tf.int64),
    'scab': tf.io.FixedLenFeature([], tf.int64),
    'healthy': tf.io.FixedLenFeature([], tf.int64)
}


def read_tfrecord(example, labeled=True):
    example = tf.io.parse_single_example(example, FEATURE_MAP)
    image = decode_image(example['image'])
    if labeled:
        label = [tf.cast(example[x], tf.float32) for x in CLASSES]
    else:
        label = example['image_name']
    return image, label


def create_dataset(filenames, labeled=True, ignore_order=False, shuffle=False,
                   repeat=False, cache=False, distribute=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

    if ignore_order:
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)

    dataset = dataset.map(lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(2048, seed=SEED)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE)

    if distribute:
        dataset = STRATEGY.experimental_distribute_dataset(dataset)

    return dataset


def solve_dataset(index, train_ids, val_ids):
    train_filenames = []
    val_filenames = []

    for i in train_ids:
        for k in range(len(GCS_PATH_TRAIN_AUG)):
            train_filenames += tf.io.gfile.glob(os.path.join(GCS_PATH_TRAIN_AUG[k], f"fold_{i}", '*.tfrec'))

    for i in val_ids:
        val_filenames += tf.io.gfile.glob(os.path.join(GCS_PATH_TEST_NO_AUG, f"fold_{i}", '*.tfrec'))

    np.random.shuffle(train_filenames)
    train_dataset = create_dataset(train_filenames, shuffle=True, repeat=True)
    val_dataset = create_dataset(val_filenames, cache=True)

    train_size = count_data_items(train_filenames)
    val_size = count_data_items(val_filenames)

    return dict(
        index=index,
        x=train_dataset,
        train_size=train_size,
        validation_data=val_dataset,
        val_size=val_size,
        steps_per_epoch=train_size // BATCH_SIZE,
        validation_steps=val_size // BATCH_SIZE,
    )


def k_fold_data_generator():
    k_folds = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for i, (train_ids, val_ids) in enumerate(k_folds.split(list(range(FOLDS)))):
        if i not in USE_FOLDS:
            continue
        yield solve_dataset(i, train_ids, val_ids)


def solve_data_generator():
    global DATA_GENERATOR
    if DATA_GENERATOR == "k_fold_data_generator":
        return k_fold_data_generator()
