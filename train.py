import os

from utils import retrieve_args_global_dict, solve_folder_path


def train():
    global MODEL
    global DATA_GENERATOR
    global MODEL_SAVE_DIR

    k_fold_history = []

    for d in DATA_GENERATOR:
        print(f"========== Fold {d['index']} ==========")

        history = MODEL.fit(**retrieve_args_global_dict(MODEL.fit, d))

        folder = solve_folder_path(MODEL_SAVE_DIR)

        MODEL.save_weights(os.path.join(folder, f"model_{d['index']}.h5"))

        k_fold_history.append(history.history)

    return k_fold_history
