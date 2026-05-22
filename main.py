import numpy as np
import cv2 as cv
import numpy
import torch
import torch.nn.functional as F
# from facenet_pytorch import MTCNN
from mtcnn import MTCNN

face_db = {}
THRESHOLD = 0.75 # TODO adjust as needed

# switch to gpu and load model
device = "cuda" if torch.cuda.is_available() else "cpu"

# MTCNN initialisation
# reference: https://github.com/ipazc/mtcnn
mtcnn = MTCNN()
model = torch.jit.load("models/artifacts/metric_learning_scripted.pt",map_location=device)

model.eval()
model.to(device)

def preprocess(face_img):
    face = cv.resize(face_img, (224, 224))
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5  # if trained with normalisation
    face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(device)

    return face


def get_embedding(face_img):
    x = preprocess(face_img)
    with torch.no_grad():
        emb = model(x)
    emb = F.normalize(emb, p=2.0, dim=1)

    return emb.squeeze(0)


# def get_embedding(face_img):
#     x = preprocess(face_img)
#     with torch.no_grad():
#         emb = model(x)
#     emb = F.normalize(emb, p=2.0, dim=1)
#     return emb.squeeze(0)


def extract_face(frame):

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    detections = mtcnn.detect_faces(rgb)
    if len(detections) == 0:
        return None, None

    # take first detected face
    detection = detections[0]
    x, y, w, h = detection["box"]

    x = max(0, x)
    y = max(0, y)
    face = rgb[y:y+h, x:x+w]
    if face.size == 0:
        return None, None

    return face, (x, y, w, h)


def register(name, frame):

    face, _ = extract_face(frame)

    if face is None:
        print("No face detected")
        return

    emb = get_embedding(face)

    if name not in face_db:
        face_db[name] = []

    face_db[name].append(emb)
    print(f"{name} registered")


def cosine_similarity(a, b):
    return torch.dot(a, b).item()


# def cosine_similarity_score(emb_a, emb_b):
#     return F.cosine_similarity(emb_a, emb_b, dim=1)


def recognition(frame):

    face, box = extract_face(frame)

    if face is None:
        return "No face", 0.0, None

    emb = get_embedding(face)
    best_name = "unknown"
    best_score = -1
    for name, embeddings in face_db.items():
        scores = [cosine_similarity(emb, e) for e in embeddings]
        score = max(scores)

        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= THRESHOLD:
        return best_name, best_score, box
    else:
        return "unknown", best_score, box


def main():

    print("test main")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press R to register current face")
    print("Press Q to quit")

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        name, score, box = recognition(frame)

        if box is not None:
            x, y, w, h = box
            cv.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0),2)

        cv.putText(frame,f"{name} ({score:.2f})",(50, 50), cv.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)

        # Our operations on the frame come here
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        # cv.imshow('frame', gray)
        # if cv.waitKey(1) == ord('q'):
        #     break

        cv.imshow("Face Recognition Interface", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("r"): # TODO implement registering function
            username = input("Enter name: ")
            register(username, frame)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()