GCS_PATH_TEST_NO_AUG = "gs://kds-3599f5f7526c2ea6ccd64f8ece7664680527d5f595413f935834d418"

GCS_PATH_TRAIN_AUG = [
    'gs://kds-a8e384195fb92611c29499c8788ad9fe1684449e469b5d4d3d5598e5',
    'gs://kds-50a8ace68dd5bd984d7d4a1e8426ef344a346580b23d636fa7533ff0',
    'gs://kds-0164bd7e9cc4a6dad070e92aff36279aac942a4a2949cec8a95dda5f',
    'gs://kds-ffb07abeeb2f1472d5d9372c3b6430af1e9849d671969eacb6fff939'
]

IMSIZE = 600

CLASSES = [
    'complex',
    'frog_eye_leaf_spot',
    'powdery_mildew',
    'rust',
    'scab'
]

NUM_CLASSES = len(CLASSES)

SEED = 1712

EPOCHS = 100

STEPS_PER_EPOCH = 115

FOLDS = 5

USE_FOLDS = [1, 2, 3, 4, 5]

PATIENCE = [5, 2]

FACTOR = .1

MIN_LR = 1e-8

VERBOSE = 1

MIXED_PRECISION = "mixed_float16"

# seed_everything(SEED)
# TPU, STRATEGY = solve_hardware(mixed_precision=MIXED_PRECISION)

# if TPU is None:
#     BATCH_SIZE = 32
# else:
#     BATCH_SIZE = 256

DATA_GENERATOR = "k_fold_data_generator"
