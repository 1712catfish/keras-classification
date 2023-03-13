import tensorflow as tf
from keras_cv_attention_models import *


def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2, ]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01,
                       eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads


class AGCModel(tf.keras.Model):
    def __init__(self, model, clip_factor=0.01, eps=1e-3):
        super(AGCModel, self).__init__()
        self.model = model
        self.clip_factor = clip_factor
        self.eps = eps

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        agc_gradients = adaptive_clip_grad(trainable_params, gradients,
                                           clip_factor=self.clip_factor, eps=self.eps)
        self.optimizer.apply_gradients(zip(agc_gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self.model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class SAMModelWithAGC(tf.keras.Model):
    """
    Arxiv article: [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf)
    Implementation by: [Keras SAM (Sharpness-Aware Minimization)](https://qiita.com/T-STAR/items/8c3afe3a116a8fc08429)
    Usage is same with `keras.models.Model`: `model = SAMModel(inputs, outputs, rho=sam_rho, name=name)`
    """

    def __init__(self, *args, rho=0.05, clip_factor=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        import tensorflow as tf

        self.rho = tf.constant(rho, dtype=tf.float32)
        self.tf = tf
        self.clip_factor = clip_factor

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # 1st step
        with self.tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        norm = self.tf.linalg.global_norm(gradients)
        scale = self.rho / (norm + 1e-12)
        e_w_list = []

        # adaptive gradient clipping
        gradients = adaptive_clip_grad(trainable_vars, gradients,
                                       clip_factor=self.clip_factor, eps=1e-7)

        for v, grad in zip(trainable_vars, gradients):
            e_w = grad * scale
            v.assign_add(e_w)
            e_w_list.append(e_w)

        # 2nd step
        with self.tf.GradientTape() as tape:
            y_pred_adv = self(x, training=True)
            loss_adv = self.compiled_loss(y, y_pred_adv, sample_weight=sample_weight, regularization_losses=self.losses)
        gradients_adv = tape.gradient(loss_adv, trainable_vars)
        for v, e_w in zip(trainable_vars, e_w_list):
            v.assign_sub(e_w)

        # adaptive gradient clipping
        agc_gradients = adaptive_clip_grad(trainable_vars, gradients_adv,
                                           clip_factor=self.clip_factor, eps=1e-7)

        # optimize
        self.optimizer.apply_gradients(zip(agc_gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


def solve_training_protocol(inputs, outputs, training_protocol="SAM+AGC"):
    if not training_protocol:
        print("Using vanilla training protocol")
        return tf.keras.Model(inputs, outputs)

    if training_protocol.upper() == "SAM":
        print("Training protocol: SAM")

        return model_surgery.SAMModel(inputs=inputs, outputs=outputs)

    if training_protocol.upper() == "AGC":
        print("Training protocol: AGC")

        return AGCModel(tf.keras.Model(inputs=inputs, outputs=outputs))

    if training_protocol.upper() == "SAM+AGC":
        print("Training protocol: SAM + AGC")
        return SAMModelWithAGC(inputs=inputs, outputs=outputs)

    print("Using vanilla training protocol")
    return tf.keras.Model(inputs=inputs, outputs=outputs)


