import tensorflow as tf
import datetime


def get_batch_config(
        *,
        strategy,
        n_train_samples,
        n_val_samples,
        batch_size_per_replica,
        batches_per_update
):
    # The number of examples for which the training procedure running on a single
    # replica will compute the gradients in order to accumulate them.
    BATCH_SIZE_PER_REPLICA = batch_size_per_replica

    # The total number of examples for which the training procedure
    # will compute the gradients in order to accumulate them.
    # This is also used for validation step.
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # Accumulate `BATCHES_PER_UPDATE` of gradients before updating the model's parameters.
    BATCHES_PER_UPDATE = batches_per_update

    # The number of examples for which the training procedure will update the model's parameters once.
    # This is the `effective` batch size, which will be used in tf.data.Dataset.
    UPDATE_SIZE = BATCH_SIZE * BATCHES_PER_UPDATE

    # The number of parameter updates in 1 epoch
    UPDATES_PER_EPOCH = n_train_samples // UPDATE_SIZE

    # The number of batches for a validation step.
    VALID_BATCHES_PER_EPOCH = n_val_samples // BATCH_SIZE

    return dict(
        BATCH_SIZE_PER_REPLICA=BATCH_SIZE_PER_REPLICA,
        BATCH_SIZE=BATCH_SIZE,
        BATCHES_PER_UPDATE=BATCHES_PER_UPDATE,
        UPDATE_SIZE=UPDATE_SIZE,
        UPDATES_PER_EPOCH=UPDATES_PER_EPOCH,
        VALID_BATCHES_PER_EPOCH=VALID_BATCHES_PER_EPOCH
    )


def set_routines(
        *,
        strategy, model, loss_fn, optimizer,
        update_size, train_accuracy, train_loss,
        batch_per_update, batch_size_per_replica,
        updates_per_epoch, valid_batches_per_epoch,
        valid_accuracy, valid_loss,
        batch_size
):
    with strategy.scope():

        def train_step_1_forward_backward(images, labels):
            """
            The procedure to be run on each replica that computes gradients by feedforward and backpropagation.

            The `images` and `labels` must have batch dimension equal to `BATCH_SIZE_PER_REPLICA`.
            """

            with tf.GradientTape() as tape:
                probabilities = model(images, training=True)
                per_example_loss = loss_fn(labels, probabilities)
                loss = tf.math.reduce_sum(per_example_loss) / update_size

            grads = tape.gradient(loss, model.trainable_variables)

            # update metrics
            train_accuracy.update_state(labels, probabilities)
            train_loss.update_state(loss)

            return grads

        def train_step_1_update(images, labels):
            """
            The procedure to be run on each replica that computes gradients
            in an accumulated way and updates the model's parameter
            (once the accumulated gradients on each replica are synced across the replicas by summing them).

            The `images` and `labels` must have batch dimension equal to `BATCHES_PER_UPDATE * BATCH_SIZE_PER_REPLICA`.
            They are splitted into `BATCHES_PER_UPDATE` parts, and each part (which has batch dimension `BATCH_SIZE_PER_REPLICA`)
            is sent to `train_step_1_forward_backward()` to compute the loss and gradients, then the gradients are added to `accumulated_grads`.

            *** Implementation detail:

                In order to split `images` and `labels` into smaller portions, the easiest way is to do something like `images[start_idx:end_idx]`
                with `start_idx = BATCHES_PER_UPDATE * batch_idx` and `end_idx = start_idx + BATCHES_PER_UPDATE`.

                However, this gives the following error:

                    Compilation failure: XLA can't deduce compile time constant output shape for strided slice: [?,512,512,3],
                    output shape must be a compile-time constant.

                Similar error is thrown for `tf.gather(images, tf.range(start_idx, end_idx))`.

                If we use the trick like `images[:BATCH_SIZE_PER_REPLICA]` at the beginning inside the `for` loop
                and modify `images` by `images = images[BATCH_SIZE_PER_REPLICA:]` at the end inside the `for` loop,
                we get another error:

                    "images" has shape (256, 512, 512, 3) before the loop, but shape (240, 512, 512, 3) after one iteration.
                    TensorFlow control flow requires it stays the same or be more specific.

                The solution given here is to do the following trick:

                    for batch_idx in tf.range(BATCHES_PER_UPDATE):

                        ...

                        small_images = images[:BATCH_SIZE_PER_REPLICA]

                        ...

                        tf.concat([images[BATCH_SIZE_PER_REPLICA:], small_images], axis=0)

                The idea is to take the first `BATCHES_PER_UPDATE` examples from `images` and pass them to `train_step_1_forward_backward`, and then
                move this portion to the end of `images`, so `images` always has the same shape, although the content is modified.

            Args:

                images: tf.Tensor with shape [BATCHES_PER_UPDATE * BATCH_SIZE_PER_REPLICA , height, width, depth]

                labels: tf.Tensor with shape [BATCHES_PER_UPDATE * BATCH_SIZE_PER_REPLICA]
            """

            accumulated_grads = [tf.zeros_like(var, dtype=tf.float32) for var in model.trainable_variables]

            for batch_idx in tf.range(batch_per_update):
                # This is not working. (Error: output shape must be a compile-time constant.)
                # start_idx = BATCHES_PER_UPDATE * batch_idx
                # end_idx = start_idx + BATCHES_PER_UPDATE
                # small_images = images[start_idx:end_idx]
                # small_labels = labels[start_idx:end_idx]

                # Take the 1st `BATCH_SIZE_PER_REPLICA` examples.
                small_images = images[:batch_size_per_replica]
                small_labels = labels[:batch_size_per_replica]

                grads = train_step_1_forward_backward(small_images, small_labels)

                accumulated_grads = [x + y for x, y in zip(accumulated_grads, grads)]

                # Move the leading part to the end, so the shape is not changed.
                images = tf.concat([images[batch_size_per_replica:], small_images], axis=0)
                labels = tf.concat([labels[batch_size_per_replica:], small_labels], axis=0)

            # Update the model's parameters.
            optimizer.apply_gradients(zip(accumulated_grads, model.trainable_variables))

        @tf.function
        def train_step_1_epoch(data_iter):

            for _ in tf.range(updates_per_epoch):
                strategy.experimental_run_v2(train_step_1_update, next(data_iter))

        @tf.function
        def valid_step(data_iter):

            def valid_step_fn(images, labels):
                probabilities = model(images, training=False)
                per_example_loss = loss_fn(labels, probabilities)
                loss = tf.math.reduce_sum(per_example_loss) / batch_size

                # update metrics
                valid_accuracy.update_state(labels, probabilities)
                valid_loss.update_state(loss)

            for _ in tf.range(valid_batches_per_epoch):
                strategy.experimental_run_v2(valid_step_fn, next(data_iter))

    return train_step_1_epoch, valid_step


def train(
        *,
        strategy, model, loss_fn, optimizer,
        train_ds, val_ds, n_train_samples, n_val_samples,
        train_accuracy, train_loss, valid_accuracy, valid_loss,
        batch_size_per_replica, batches_per_update,
        steps, steps_per_epoch=None
):
    batch_configuration = get_batch_config(
        strategy=strategy,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        batch_size_per_replica=batch_size_per_replica,
        batches_per_update=batches_per_update,
    )

    UPDATES_PER_EPOCH = batch_configuration["updates_per_epoch"]
    VALID_BATCHES_PER_EPOCH = batch_configuration["valid_batches_per_epoch"]

    train_step_1_epoch, valid_step = set_routines(
        strategy=strategy,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_accuracy=train_accuracy,
        train_loss=train_loss,
        valid_accuracy=valid_accuracy,
        valid_loss=valid_loss,
        **batch_configuration,
    )

    for step_i in range(steps):
        s = datetime.datetime.now()

        train_step_1_epoch(train_ds)

        loss = train_loss.result() / UPDATES_PER_EPOCH
        acc = train_accuracy.result()

        print("epoch: {}".format(step_i + 1))

        print("train loss: {}".format(loss))
        print("train accuracy: {}".format(acc))

        train_loss.reset_states()
        train_accuracy.reset_states()

        e = datetime.datetime.now()
        print("elapsed: {}".format((e - s).total_seconds()))

        valid_step(val_ds)

        val_loss = valid_loss.result() / VALID_BATCHES_PER_EPOCH
        val_acc = valid_accuracy.result()

        print("valid loss: {}".format(val_loss))
        print("valid accuracy: {}".format(val_acc))

        valid_loss.reset_states()
        valid_accuracy.reset_states()

        print("-" * 80)
