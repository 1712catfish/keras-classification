from utils import list_valid_args, list_global_variables, args_global_dict


def train():
    global MODEL
    global DATA_GENERATOR

    k_fold_history = []

    for d in DATA_GENERATOR:
        print(f"========== Fold {d['index']} ==========")

        history = MODEL.fit(**args_global_dict(d))

        MODEL.save_weights(f"model_{d['index']}.h5")
        k_fold_history.append(history.history)

    return k_fold_history
