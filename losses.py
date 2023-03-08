import tensorflow as tf


def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


def stochastic_loss(losses):
    def res(y_true, y_pred):
        rand = tf.random.uniform([])
        for i in range(len(losses)):
            if rand < float(i + 1) / float(len(losses)):
                return losses[i](y_true, y_pred)

    return res


# Utilities for loss implementation.


def get_logsumexp_loss(states, temperature):
    scores = tf.matmul(states, states, transpose_b=True)  # (bsz, bsz)
    bias = tf.math.log(tf.cast(tf.shape(states)[1], tf.float32))  # a constant
    return tf.reduce_mean(
        tf.math.reduce_logsumexp(scores / temperature, 1) - bias)


def sort(x):
    """Returns the matrix x where each row is sorted (ascending)."""
    xshape = tf.shape(x)
    rank = tf.reduce_sum(
        tf.cast(tf.expand_dims(x, 2) > tf.expand_dims(x, 1), tf.int32), axis=2)
    rank_inv = tf.einsum(
        'dbc,c->db',
        tf.transpose(tf.cast(tf.one_hot(rank, xshape[1]), tf.float32), [0, 2, 1]),
        tf.range(xshape[1], dtype='float32'))  # (dim, bsz)
    x = tf.gather(x, tf.cast(rank_inv, tf.int32), axis=-1, batch_dims=-1)
    return x


def get_swd_loss(states, rand_w, prior='normal', stddev=1., hidden_norm=True):
    states_shape = tf.shape(states)
    states = tf.matmul(states, rand_w)
    states_t = sort(tf.transpose(states))  # (dim, bsz)

    if prior == 'normal':
        states_prior = tf.random.normal(states_shape, mean=0, stddev=stddev)
    elif prior == 'uniform':
        states_prior = tf.random.uniform(states_shape, -stddev, stddev)
    else:
        raise ValueError('Unknown prior {}'.format(prior))
    if hidden_norm:
        states_prior = tf.math.l2_normalize(states_prior, -1)
    states_prior = tf.matmul(states_prior, rand_w)
    states_prior_t = sort(tf.transpose(states_prior))  # (dim, bsz)

    return tf.reduce_mean((states_prior_t - states_t) ** 2)


def generalized_contrastive_loss(
        hidden1,
        hidden2,
        lambda_weight=1.0,
        temperature=1.0,
        dist='normal',
        hidden_norm=True,
        loss_scaling=1.0):
    """Generalized contrastive loss.

    Both hidden1 and hidden2 should have shape of (n, d).

    Configurations to get following losses:
    * decoupled NT-Xent loss: set dist='logsumexp', hidden_norm=True
    * SWD with normal distribution: set dist='normal', hidden_norm=False
    * SWD with uniform hypersphere: set dist='normal', hidden_norm=True
    * SWD with uniform hypercube: set dist='uniform', hidden_norm=False
    """
    hidden_dim = hidden1.shape[-1]  # get hidden dimension
    if hidden_norm:
        hidden1 = tf.math.l2_normalize(hidden1, -1)
        hidden2 = tf.math.l2_normalize(hidden2, -1)
    loss_align = tf.reduce_mean((hidden1 - hidden2) ** 2) / 2.
    hiddens = tf.concat([hidden1, hidden2], 0)
    if dist == 'logsumexp':
        loss_dist_match = get_logsumexp_loss(hiddens, temperature)
    else:
        initializer = tf.keras.initializers.Orthogonal()
        rand_w = initializer([hidden_dim, hidden_dim])
        loss_dist_match = get_swd_loss(hiddens, rand_w,
                                       prior=dist,
                                       hidden_norm=hidden_norm)
    return loss_scaling * (loss_align + lambda_weight * loss_dist_match)


def PseudoContrastiveLoss(
        margin=1.0,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
):

    contrastive_loss = tfa.losses.ContrastiveLoss(margin=margin, reduction=reduction, )

    cossim = tf.keras.losses.CosineSimilarity(axis=-1, reduction=reduction, )

    def pseudo_contrastive_loss(y_true, y_pred, sample_weight=None):
        indices_1 = tf.random.uniform([tf.shape(y_pred)[0]], minval=0, maxval=tf.shape(y_pred)[0], dtype=tf.int32)
        indices_2 = tf.random.uniform([tf.shape(y_pred)[0]], minval=0, maxval=tf.shape(y_pred)[0], dtype=tf.int32)

        y_pred_1 = tf.gather(y_pred, indices_1)
        y_pred_2 = tf.gather(y_pred, indices_2)

        y_true_1 = tf.gather(y_true, indices_1)
        y_true_2 = tf.gather(y_true, indices_2)

        y_sim = cossim(y_pred_1, y_pred_2)
        y_sim_true = cossim(y_true_1, y_true_2)

        loss = contrastive_loss(y_sim_true, y_sim, sample_weight=sample_weight)

        # loss = contrastive_loss(
        #     y_pred_1, y_pred_2,
        #     lambda_weight=1.0, temperature=1.0, dist='normal', hidden_norm=True, loss_scaling=1.0
        # )

        return loss

    return pseudo_contrastive_loss
