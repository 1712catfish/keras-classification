class LRReset(tf.keras.callbacks.Callback):
    def __init__(self, epoch_freq=10, lr=1e-3):
        super().__init__()
        self.epoch_freq = epoch_freq
        self.lr = lr

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_freq == 0:
            K.set_value(self.model.optimizer.lr, self.lr)
        print("End epoch {} of training; reset lr to {}".format(epoch, self.lr))