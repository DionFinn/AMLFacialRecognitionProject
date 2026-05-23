import numpy as np
import cv2 as cv
import tensorflow as tf


def predict_liveness(model, frame, threshold=0.7):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    input_frame = cv.resize(frame, (224, 224))
    input_frame = np.expand_dims(input_frame, axis=0)

    pred = model.predict(input_frame, verbose=0)[0][0]

    if pred >= threshold:
        return "Spoof", pred
    else:
        return "Real", pred