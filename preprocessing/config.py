import os

DATASET_PATH = 'data/11-785-fall-20-homework-2-part-2/'
CLASSIFICATION_PATH = os.path.join(DATASET_PATH, 'classification_data/')
VERIFICATION_PATH = os.path.join(DATASET_PATH, 'verification_data/')
TRAIN_PATH = os.path.join(CLASSIFICATION_PATH, 'train_data/')
TEST_PATH = os.path.join(CLASSIFICATION_PATH, 'test_data/')
VAL_PATH = os.path.join(CLASSIFICATION_PATH, 'val_data/')
VERIFICATION_PAIRS_VAL = os.path.join(DATASET_PATH, 'verification_pairs_val.txt')

IMG_SIZE = (342, 342)

paths = {
    "Dataset": DATASET_PATH,
    "Classification": CLASSIFICATION_PATH,
    "Verification": VERIFICATION_PATH,
    "Train Path": TRAIN_PATH,
    "Test Path": TEST_PATH,
    "Val Path": VAL_PATH,
    "Verification Pairs val": VERIFICATION_PAIRS_VAL
}
