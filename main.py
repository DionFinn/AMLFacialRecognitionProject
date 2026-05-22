import numpy as np
import cv2 as cv
import tensorflow as tf

from model.antispoof_preprocess import predict_liveness

antispood_raw = tf.keras.models.load_model("./model/models/antispoof_raw.keras")
antispoof_transfer = tf.keras.models.load_model("./model/models/antispoof_transfer.keras")

if not antispood_raw:
    ValueError("model not found")
if not antispoof_transfer:
    ValueError("model not found")


def main():
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
    
        ret, frame = cap.read()
    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        label, score = predict_liveness(antispoof_transfer, frame, threshold=0.7)

        cv.putText(
            frame,
            f"{label}: {score:.2f}",
            (30, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()