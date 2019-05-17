'''

Author: nyismaw
Constants for Casava image files as a part of an in-class kaggle competition
https://www.kaggle.com/c/cassava-disease/

'''

CBB_IMAGES_DIR            = "data_sets/train/cbb/*"
CBSD_IMAGES_DIR           = "data_sets/train/cbsd/*"
CGM_IMAGES_DIR            = "data_sets/train/cgm/*"
CMD_IMAGES_DIR            = "data_sets/train/cmd/*"
HEALTHY_IMAGES_DIR        = "data_sets/train/healthy/*"

CBB_LABEL                 = 0
CBSD_LABEL                = 1
CGM_LABEL                 = 2
CMD_LABEL                 = 3
HEALTHY_LABEL             = 4

TRAIN_DATA_PATH           = "data_sets/train_data.npy"
TRAIN_LABEL_PATH          = "data_sets/train_label.npy"
VAL_DATA_PATH             = "data_sets/val.npy"
VAL_LABEL_PATH            = "data_sets/val_label.npy"

TRAIN_VAL_SPLIT           = 0.8

AVERAGE_HEIGHT            = 574
AVERAGE_WIDTH             = 606

EPOCH                     = 10
LEARNING_RATE             = 1e-3
WEIGHT_DECAY              = 1e-6
SAVE_MODEL_INTERVAL       = 10
PRINT_TRAIN_INFO_INTERVAL = 10
