from utils import list_valid_args, retrieve_global_variables


def train(model, data_generator):
    k_fold_history = []
    for d in data_generator:
        print(f"========== Fold {d['index']} ==========")

        global_settings = retrieve_global_variables()

        args_dict = dict()
        for k in set(list_valid_args(model.fit)).intersection((set(global_settings).union(set(d.keys())))):
            args_dict[k] = global_settings.get(k, d.get(k, None))

        print(1)

        history = model.fit(**args_dict)

        model.save_weights(f"model_{d['index']}.h5")
        k_fold_history.append(history.history)

    return model, k_fold_history
