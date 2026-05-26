import cv2 as cv
import numpy as np 
import pandas as pd
import os

DATASET_PATH = 'data/' # ! alter path according to your structure -> temporary fix and must decide on a common structure afterwards

CLASSIFICATION_PATH = os.path.join(DATASET_PATH, 'classification_data/')
VERIFICATION_PATH = os.path.join(DATASET_PATH, 'verification_data/')

TRAIN_PATH = os.path.join(CLASSIFICATION_PATH, 'train_data/')
TEST_PATH = os.path.join(CLASSIFICATION_PATH, 'test_data/')
VAL_PATH = os.path.join(CLASSIFICATION_PATH, 'val_data/')

IMG_SIZE = (112, 112) # ! CHANGED FROM 224X224

paths = {
    "Dataset": DATASET_PATH,
    "Classification": CLASSIFICATION_PATH,
    "Verification": VERIFICATION_PATH,
    "Train Path": TRAIN_PATH,
    "Test Path": TEST_PATH,
    "Val Path": VAL_PATH
}

for name, path in paths.items():
    if os.path.exists(path):
        continue
    else:
        print(f"path {name} not found with path {path}")

def preprocess_image(image_path, img_size=IMG_SIZE):
    image = cv.imread(image_path)
    if image is None:
        return f"image not found with path {image_path}"

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, img_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    # img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

    return image




