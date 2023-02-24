from utils import list_valid_args, retrieve_global_variables


def train():
    k_fold_history = []
    for d in DATA_GENERATOR:
        print(f"========== Fold {d['index']} ==========")

        global_settings = retrieve_global_variables()

        args_dict = dict()
        for k in set(list_valid_args(MODEL.fit)).intersection((set(global_settings).union(set(d.keys())))):
            args_dict[k] = global_settings.get(k, d.get(k, None))

        history = MODEL.fit(**args_dict)

        MODEL.save_weights(f"model_{d['index']}.h5")
        k_fold_history.append(history.history)

    return k_fold_history