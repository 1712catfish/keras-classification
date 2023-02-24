import inspect
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

"""Argument parsing"""


def list_global_variables(condition=None, filter_underscore=True, **kwargs):
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


def args_global_dict(func, d):
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
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
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