import inspect

import numpy as np
import tensorflow as tf

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


"""Hardware configuration"""


def solve_hardware(mixed_precision="mixed_float16"):
    tf.keras.mixed_precision.set_global_policy(mixed_precision)

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
    np.random.set_state(seed)
