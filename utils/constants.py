import os


# Locations of the data you downloaded from the Kaggle competition.
RAW_METADATA_PATH = os.path.join('downloaded-raw-data/train_ship_segmentations_v2.csv')
RAW_TRAIN_IMAGES_DIR = os.path.join('downloaded-raw-data/train_v2/')

# Locations of the data you preprocessed.
LOCAL_DATA_DIR = os.path.join('data/')
TRAIN_IMAGES_DIR = os.path.join(LOCAL_DATA_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(LOCAL_DATA_DIR, "train", "masks")
TEST_IMAGES_DIR = os.path.join(LOCAL_DATA_DIR, "test", "images")
TEST_MASKS_DIR = os.path.join(LOCAL_DATA_DIR, "test", "masks")

# Locations of the metadata you preprocessed.
TRAIN_METADATA_PATH = os.path.join(LOCAL_DATA_DIR, "train.csv")
TEST_METADATA_PATH = os.path.join(LOCAL_DATA_DIR, "test.csv")

# Image and mask shapes. It is not recommended to change INITIAL_MASK_SHAPE.
IMAGE_SHAPE_3D = (192, 192, 3)
IMAGE_SHAPE_2D = IMAGE_SHAPE_3D[:2]
INITIAL_MASK_SHAPE = (768, 768)

# Training parameters
EPOCHS = 10
BATCH_SIZE = 4

# Model parameters
MODEL_PATH = os.path.join('model/models/model-2024-03-01.h5')
RESULTS_DIR = os.path.join('results/')

# Default test image path for segmentation.py
DEFAULT_TEST_IMAGE_PATH = os.path.join('data/test/images/0a1a58833.jpg')
