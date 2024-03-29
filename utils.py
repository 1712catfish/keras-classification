import inspect
import os

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

"""Argument parsing"""


def list_global_variables(condition=None, filter_underscore=True, **kwargs):
    global globals

    def cond(k):
        if k == "list_global_variables":
            return False
        if filter_underscore and k.startswith('_'):
            return False
        if condition is not None:
            return condition(k)
        return True

    return list(filter(cond, globals().keys()))


def list_global_constants(condition=None, filter_underscore=True, **kwargs):
    upper = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890')

    def cond(k):
        if condition is not None and not condition(k):
            return False
        if not all(c in upper for c in k):
            return False
        return True

    return list_global_variables(condition=cond, filter_underscore=filter_underscore, **kwargs)


def filter_dict_by_value(condition, d):
    return {k: v for k, v in d.items() if condition(v)}


def filter_dict_by_key(condition, d):
    return {k: v for k, v in d.items() if condition(k)}


def list_valid_args(func):
    return list(inspect.signature(func).parameters.keys())


def retrieve_global_variables(keys=None, lower=True, **kwargs):
    if keys is not None:
        d = {k: eval(k) for k in keys}
    else:
        d = {k: eval(k) for k in list_global_variables(**kwargs)}
    return {k.lower(): v for k, v in d.items()}


def retrieve_global_valid_constants(func):
    valid_args = list_valid_args(func)
    constants = list_global_constants(condition=lambda k: not callable(eval(k)))
    keys = [k for k in constants if k.lower() in valid_args]
    return retrieve_global_variables(keys)


def resolve_global_constants(d, **kwargs):
    constants = list_global_constants(**kwargs)
    for k in intersect([k.lower() for k in constants], d.keys()):
        d[k] = eval(k.upper())
    return d


def dict_update(d, d_new):
    for k in set(d).intersection(set(d_new)):
        d[k] = d_new
    return d


def dict_key_lower(d):
    return {k.lower(): v for k, v in d.items()}


def dict_key_upper(d):
    return {k.upper(): v for k, v in d.items()}


def intersect(*keys):
    if len(keys) == 0:
        return []
    if len(keys) == 1:
        return keys
    s = set(keys[0])
    for key in keys[1:]:
        s = s.intersection(set(key))
    return s


def retrieve_args_global_dict(func, d):
    global globals

    global_settings = list_global_variables()
    settings = set(d).union(set([k.lower() for k in global_settings]))
    args_dict = dict()
    for k in set(list_valid_args(func)).intersection(settings):
        if k.upper() in global_settings:
            args_dict[k] = eval(k.upper())
        elif d.get(k, None) is not None:
            args_dict[k] = d[k]
    return args_dict


"""Hardware configuration"""


def solve_hardware():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print('Running on TPUv3-8')
    except:
        tpu = None
        strategy = tf.distribute.get_strategy()
        print('Running on GPU with mixed precision')

    batch_size = 16 * strategy.num_replicas_in_sync

    print('Number of replicas:', strategy.num_replicas_in_sync)
    print('Batch size: %.i' % batch_size)

    return tpu, strategy


def seed_everything(seed):
    tf.random.set_seed(seed)
    # np.random.set_state(seed)


def plot_history_metric(history, metric, f_best=np.argmax, ylim=None, yscale=None, yticks=None):
    # https://www.kaggle.com/code/markwijkhuizen/rsna-convnextv2-training-tensorflow-tpu?scriptVersionId=116484001
    plt.figure(figsize=(20, 10))

    values = history[metric]
    N_EPOCHS = len(values)
    val = 'val' in ''.join(history.keys())
    # Epoch Ticks
    if N_EPOCHS <= 20:
        x = np.arange(1, N_EPOCHS + 1)
    else:
        x = [1, 5] + [10 + 5 * idx for idx in range((N_EPOCHS - 10) // 5 + 1)]

    x_ticks = np.arange(1, N_EPOCHS + 1)

    # Validation
    if val:
        val_values = history[f'val_{metric}']
        val_argmin = f_best(val_values)
        plt.plot(x_ticks, val_values, label=f'val')

    # summarize history for accuracy
    plt.plot(x_ticks, values, label=f'train')
    argmin = f_best(values)
    plt.scatter(argmin + 1, values[argmin], color='red', s=75, marker='o', label=f'train_best')
    if val:
        plt.scatter(val_argmin + 1, val_values[val_argmin], color='purple', s=75, marker='o', label=f'val_best')

    plt.title(f'Model {metric}', fontsize=24, pad=10)
    plt.ylabel(metric, fontsize=20, labelpad=10)

    if ylim:
        plt.ylim(ylim)

    if yscale is not None:
        plt.yscale(yscale)

    if yticks is not None:
        plt.yticks(yticks, fontsize=16)

    plt.xlabel('epoch', fontsize=20, labelpad=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.xticks(x, fontsize=16)  # set tick step to 1 and let x axis start at 1
    plt.yticks(fontsize=16)

    plt.legend(prop={'size': 10})
    plt.grid()
    plt.show()


def solve_folder_path(base):
    if not os.path.exists(base):
        os.makedirs(base)
        return base
    for i in range(1000):
        folder = os.path.join(base, f'{i:04d}')
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder


def plot_training_results_2(folder="./exp"):
    folder = solve_folder_path(folder)

    global FOLDS

    """
    Plot training results
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(32, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, FOLDS, figure=fig)

    for fold_idx in range(FOLDS):
        tmp_log_dir = os.path.join(folder, f"fold{fold_idx}_logs/version_0")
        metrics = pd.read_csv(os.path.join(tmp_log_dir, 'metrics.csv'))

        train_acc = metrics['train_f1'].dropna().reset_index(drop=True)
        valid_acc = metrics['valid_f1'].dropna().reset_index(drop=True)

        ax = fig.add_subplot(gs[0, fold_idx])
        ax.plot(train_acc, color="r", marker="o", label='train/f1')
        ax.plot(valid_acc, color="b", marker="x", label='valid/f1')
        ax.set_xlabel('Epoch', fontsize=24)
        ax.set_ylabel('F1', fontsize=24)
        ax.set_title(f'fold {fold_idx}')
        ax.legend(loc='lower right', fontsize=18)

        train_loss = metrics['train_loss'].dropna().reset_index(drop=True)
        valid_loss = metrics['valid_loss'].dropna().reset_index(drop=True)

        ax = fig.add_subplot(gs[1, fold_idx])
        ax.plot(train_loss, color="r", marker="o", label='train/loss')
        ax.plot(valid_loss, color="b", marker="x", label='valid/loss')
        ax.set_ylabel('Loss', fontsize=24)
        ax.set_xlabel('Epoch', fontsize=24)
        ax.legend(loc='upper right', fontsize=18)


def display_result(history, model, i=0):
    try:
        history = history.history
    except:
        pass

    try:
        os.mkdir('res')
    except:
        pass

    with open(f'res/report.txt', 'w+') as f:
        f.write(f"acc: {max(history['val_acc'])}\n\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.save_weights(os.path.join('res', "model.h5"))

    print(f"Acc: {100 * max(history['val_acc'])}")

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def inspect_distribution(m):
    from keract import get_activations

    image = tf.random.normal(128, 380, 380, 3, mean=0.5, stddev=0.5)

    for layer in m.layers:
        act = get_activations(
            m, image,
            layer_names=layer.name,
            output_format='simple',
            nested=True,
            auto_compile=True
        )
