import numpy as np
import cv2 as cv
import tensorflow as tf

from model.antispoof_preprocess import predict_liveness

antispoof_raw = tf.keras.models.load_model("./model/models/antispoof_raw.keras")
antispoof_transfer = tf.keras.models.load_model("./model/models/antispoof_transfer.keras")
antispoof_v3 = tf.keras.models.load_model("./model/models/antispoof_v3.keras")


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        label, score = predict_liveness(antispoof_v3, frame, threshold=0.55)

        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv.putText(frame, f"{label}: {score:.2f}", (30, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()