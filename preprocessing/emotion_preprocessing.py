import os
import cv2 as cv2
import numpy as np

DATASET_PATH = 'data/train'

label_mapping = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

X = []
y = []

for emotion in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, emotion)

    if not os.path.isdir(folder_path):
        continue

    label = label_mapping[emotion]

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (224,224))
        img = img.astype(np.float32) / 255.0

        X.append(img)
        y.append(label)

print(len(X), len(y))
print(X[0].shape, y[0])

X = np.array(X)
y = np.array(y)

print(X.shape, y.shape)

np.save('data/X.npy', X)
np.save('data/y.npy', y)