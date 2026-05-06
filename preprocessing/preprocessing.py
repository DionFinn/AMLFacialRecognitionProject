import cv2 as cv
import numpy as np
import pandas as pd
import os

from preprocessing.config import paths

def check_paths(paths):
    missing_paths = []

    for name, path in paths.items():
        if not os.path.exists(path):
            missing_paths.append((name, path))

    if len(missing_paths) > 0:
        for name, path in missing_paths:
            print(f"{name} not found with path {path}")
            return False

    print('all paths validated')
    return True

def validate_data_setup():
    if not check_paths(paths):
        raise ValueError('Data setup invalid, please check dataset paths')
    
    return True

def preprocess_image(image_path, img_size=(342, 342)):
    image = cv.imread(image_path)
    if image is None:
        return f"image not found with path {image_path}"

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, img_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    return image






