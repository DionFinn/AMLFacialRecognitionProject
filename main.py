import numpy as np
import cv2 as cv
import numpy
import torch
import torch.nn.functional as F
import json
# from facenet_pytorch import MTCNN
from mtcnn import MTCNN
import tensorflow as tf
from model.anti_spoof.antispoof_preprocess import predict_liveness
from collections import deque

try: 
    antispoof_raw = tf.keras.models.load_model("model/models/antispoof_raw.keras")
    antispoof_transfer = tf.keras.models.load_model("model/models/antispoof_transfer.keras")
    antispoof_v3 = tf.keras.models.load_model("model/models/antispoof_v3.keras")
except Exception as e:
    print(f"Model not found: ", e)

# switch to gpu and load model
device = "cuda" if torch.cuda.is_available() else "cpu"


THRESHOLD = 0.90 
face_db = {}
spoof_queue = deque(maxlen=5)

# MTCNN initialisation
# reference: https://github.com/ipazc/mtcnn
mtcnn = MTCNN()
model = torch.jit.load("model/models/metric_learning_scripted_v2.pt",map_location=device)

model.eval()
model.to(device)

def update_gallery(name, current_emb, alpha=0.90):
    """update stored face emb with live embeddings"""
    if name in face_db:
        stored_emb = face_db[name]
        # linear interpolation between old and new embedding
        updated_emb = (alpha*stored_emb) + ((1.0 - alpha)*current_emb)
        face_db[name] = F.normalize(updated_emb, p=2, dim=0)

def preprocess(face_img):
    face = cv.resize(face_img, (224, 224))
    face = face.astype(np.float32) / 255.0
    # face = (face - 0.5) / 0.5  # if trained with normalisation
    face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(device)

    return face

def get_embedding(face_img):
    x = preprocess(face_img)
    with torch.no_grad():
        emb = model(x)
    emb = F.normalize(emb, p=2.0, dim=1)

    return emb.squeeze(0)

def extract_face(frame):

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    detections = mtcnn.detect_faces(rgb)
    if len(detections) == 0:
        return None, None

    # take first detected face
    # detection = detections[0]
    detection = max(detections, key=lambda x: x["confidence"])
    if detection["confidence"] < 0.95:
        return None, None
    x, y, w, h = detection["box"]

    x = max(0, x)
    y = max(0, y)
    face = rgb[y:y+h, x:x+w]
    if face.size == 0:
        return None, None

    return face, (x, y, w, h)

def register(cap, name, samples=1): # could be 10 but less images to work weith
    embeddings = []

    print(f"Registering {name}")
    while len(embeddings) < samples:
        ret, frame = cap.read()
        if not ret:
            continue
        face, box = extract_face(frame)
        display = frame.copy()

        if box is not None:
            x, y, w, h = box
            cv.rectangle(display,(x, y),(x + w, y + h),(0, 255, 0),2)

        cv.putText(
            display,f"Capturing {len(embeddings)+1}/{samples}",(20, 40),cv.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2
        )

        cv.imshow("Face Recognition Interface", display)

        # if face is not None:
        if face is not None and box is not None:
            x, y, w, h = box
            # reject tiny faces
            if w < 100 or h < 100:
                continue

            emb = get_embedding(face)
            embeddings.append(emb)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            return

    mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
    mean_embedding = F.normalize(mean_embedding, p=2, dim=0)
    # face_db[name] = embeddings
    face_db[name] = mean_embedding
    # save_face_db()
    print(f"{name} registered successfully")

def cosine_similarity_score(emb_a, emb_b):
    return F.cosine_similarity(emb_a, emb_b, dim=0).item()

def recognition(frame):
    face, box = extract_face(frame)

    if face is None:
        return "No face", 0.0, None
    
    emb = get_embedding(face)
    best_name = "unknown"
    best_score = -1.0

    for name, ref_emb in face_db.items():
        score = cosine_similarity_score(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name
        # print(f"{name}: {score:.4f}")

    if best_score >= THRESHOLD:
        return best_name, best_score, box, emb

    return "unknown", best_score, box, emb


def main():
    print("Camera running...")

    cap = cv.VideoCapture(0) # 0 works for my camera, but you may need to change it to 1 or 2 if you have multiple cameras
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press R to register current face")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        
        # ! ======================= FACE-RECOGNITION =======================
        name, recognition_score, box, current_emb = recognition(frame)
        if box is not None:
            x, y, w, h = box
            cv.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0),2)

        cv.putText(frame,f"{name} ({recognition_score:.2f})",(50, 50), 
                   cv.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), 2)
        
        # ! ======================= ANTI-SPOOF =======================
        label, spoof_score = predict_liveness(antispoof_transfer, frame, threshold=0.55)
        spoof_queue.append(label)

        majority_label = max(set(spoof_queue), key=spoof_queue.count)

        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv.putText(frame, f"{majority_label}: {spoof_score:.2f}", (20, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # ! ======================= DYNAMIC GALLERY ADAPTATION =======================
        

        cv.imshow("Face Recognition Interface", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("r"):
            username = input("Enter name: ")
            register(cap, username)
        elif key == ord("s"):
            if name == "No face":
                print("[ERROR] No face detected. Look at the camera and try again.")
            elif name == "unknown":
                print("[ERROR] Cannot adapt template for 'unknown'. Register this person first using 'R'.")
            elif majority_label != "Real":
                print(f"[SECURITY] Adaptation blocked! Liveness state is currently: {majority_label}")
            else:
                update_gallery(name, current_emb, alpha=0.90)
                print(f"[SUCCESS] Handled appearance drift! Stored template for '{name}' updated successfully.")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()