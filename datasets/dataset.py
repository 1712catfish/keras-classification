import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def tfrecord_dataset(
        record_files,
        parse_record_fn=lambda example: example,
        batch_size=32,
        seed=1712,
        shuffle=False,
        repeat=False,
        cache=False,
        ignore_order=False,
        distribute=False,
        buffer_size=42,
        strategy=None,
):
    """Assume dataset is parsed from tfrecord files"""

    dataset = tf.data.TFRecordDataset(record_files, num_parallel_reads=tf.data.AUTOTUNE)

    dataset = dataset.map(parse_record_fn, num_parallel_calls=AUTOTUNE)

    if ignore_order:
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)

    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed=seed)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE)

    if distribute and strategy is not None:
        dataset = strategy.experimental_distribute_dataset(dataset)

    return dataset


def train_tfrecord_dataset(
        record_files,
        repeat=True,
        cache=False,
        **kwargs
):
    return tfrecord_dataset(
        record_files,
        repeat=repeat,
        cache=cache,
        **kwargs
    )


def test_tfrecord_dataset(
        record_files,
        image_size=260,
        repeat=False,
        cache=True,
        **kwargs
):
    return tfrecord_dataset(
        record_files,
        repeat=repeat,
        cache=cache,
        **kwargs
    )


def single_class_tfrec_ds(
        train_tfrec_files,
        classes,
        test_tfrec_files=None,
        train_image_size=224,
        test_image_size=260,
        batch_size=32,
        seed=1712,
        parse_record_fn="default",
):
    def default_parse_record_fn(example, image_size):
        example = tf.io.parse_single_example(example, {{
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }})

        image = tf.image.decode_jpeg(example["image"], channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.

        label = tf.one_hot(example["label"], classes)
        return image, label

    if parse_record_fn == "default":
        parse_record_fn = default_parse_record_fn

    train_ds = train_tfrecord_dataset(
        train_tfrec_files,
        image_size=train_image_size,
        parse_record_fn=lambda example: parse_record_fn(example, image_size=train_image_size),
        seed=seed,
        batch_size=batch_size,
    )

    if test_tfrec_files is None:
        return train_ds

    val_ds = test_tfrecord_dataset(
        test_tfrec_files,
        image_size=test_image_size,
        parse_record_fn=lambda example: parse_record_fn(example, image_size=test_image_size),
        seed=seed,
        batch_size=batch_size,
    )

    return train_ds, val_ds
