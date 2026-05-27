import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn
import torchvision.models as models
import tensorflow as tf
from antispoof_preprocess import predict_liveness
from emotion_innovation import load_emotion_system, predict_emotion
from mtcnn import MTCNN
import time

# session intiialisation
if "register_flag" not in st.session_state:
    st.session_state.register_flag = False
# ! DYNAMIC GALLERY ADAPTATION INNOVATION
if "adapt_flag" not in st.session_state:
    st.session_state.adapt_flag = False

# architecture layout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MTCNN globally for high-fidelity alignment
detector_mtcnn = MTCNN()

class MobileViT(nn.Module):
    def __init__(self, num_classes=4000):
        super(MobileViT, self).__init__()
        self.base_model = timm.create_model('mobilevit_xs.cvnets_in1k', pretrained=False)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        self.classifier = nn.Sequential(nn.Linear(384, num_classes))
    def features(self, x): return self.base_model.forward_features(x)
    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))

class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=4000):
        super(EfficientNetWrapper, self).__init__()
        self.base_model = models.efficientnet_b0(weights=None)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
    def features(self, x): return self.base_model.features(x)
    def avgpool(self, x): return self.base_model.avgpool(x)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.base_model.classifier(x)

# caching and optimisation
@st.cache_resource
def load_base_assets():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return transform, face_cascade

@st.cache_resource
def load_selected_model(model_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_choice == "MobileViT-XS (Transformer Hybrid)":
        model = torch.load("../model/models/MobileViT_supervised.pt", map_location=device, weights_only=False)
        embedding_dim = 384
    elif model_choice == "metric_learning":
        model = torch.load("../model/models/metric_learning_scripted_v3.pt", map_location=device, weights_only=False)
        embedding_dim = 384
    else:
        model = torch.load("../model/models/EfficientNet_supervised.pt", map_location=device, weights_only=False)
        embedding_dim = 1280
        
    model = model.to(device)
    model.eval()
    return model, device, embedding_dim

def extract_embedding(model, image_cv2, face_cascade, transform, device, model_choice):
    """Safely crops with padding and extracts the standardized embedding vector."""
    h_img, w_img, _ = image_cv2.shape
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    detections = detector_mtcnn.detect_faces(img_rgb)
    if len(detections) == 0: return None, None
    
    detection = max(detections, key=lambda x: x["confidence"])
    if detection["confidence"] < 0.90: return None, None
    
    x, y, w, h = detection["box"]
    x, y = max(0, x), max(0, y)
    w, h = min(w, w_img - x), min(h, h_img - y)
    
    keypoints = detection["keypoints"]
    left_eye_kp = keypoints["left_eye"]
    right_eye_kp = keypoints["right_eye"]
    
    screen_eyes = sorted([left_eye_kp, right_eye_kp], key=lambda e: e[0])
    left_eye_coords = screen_eyes[0]
    right_eye_coords = screen_eyes[1]
    
    dx = right_eye_coords[0] - left_eye_coords[0]
    dy = right_eye_coords[1] - left_eye_coords[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    eye_center = (int((left_eye_coords[0] + right_eye_coords[0]) / 2), 
                  int((left_eye_coords[1] + right_eye_coords[1]) / 2))
    
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    aligned_img = cv2.warpAffine(image_cv2, rotation_matrix, (w_img, h_img), flags=cv2.INTER_CUBIC)
    
    pad_w, pad_h = int(w * 0.05), int(h * 0.05)
    y1 = max(0, y - pad_h)
    y2 = min(h_img, y + h + pad_h)
    x1 = max(0, x - pad_w)
    x2 = min(w_img, x + w + pad_w)
    face_crop = aligned_img[y1:y2, x1:x2]
    
    if face_crop.size == 0: return None, None
    
    img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if model_choice == 'metric_learning':
            features = model.features(tensor)
            emb = features.flatten(1)
            emb = model.embedding(emb)
        else:
            features = model.features(tensor)
            emb = model.avgpool(features).flatten(1)
        emb = F.normalize(emb, p=2, dim=1)
        
    return emb, (x, y, w, h)

@st.cache_data
def precompute_database_embeddings(model_choice, registered_folder):
    model, device, _ = load_selected_model(model_choice)
    img_transform, face_cascade = load_base_assets()
    known_embeddings = {}
    
    if not os.path.exists(registered_folder): return known_embeddings
        
    # loop through folders per person
    for person_name in os.listdir(registered_folder):
        person_dir = os.path.join(registered_folder, person_name)
        if not os.path.isdir(person_dir): continue
        
        person_embs = []
        for file_name in os.listdir(person_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, file_name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                emb, _ = extract_embedding(model, img, face_cascade, img_transform, device, model_choice)
                if emb is not None:
                    person_embs.append(emb)
                    
        # Multi-Template Vector Space Aggregation
        if person_embs:
            mean_emb = torch.mean(torch.stack(person_embs), dim=0)
            known_embeddings[person_name] = F.normalize(mean_emb, p=2, dim=1)
                
    return known_embeddings

# ! MAIN GUI LAYOUT
st.title("Face Recognition and Verification System")

selected_engine = st.sidebar.selectbox("Active Model Engine", ("MobileViT-XS (Transformer Hybrid)", "EfficientNet-B0 (Pure CNN)", "metric_learning"))
threshold = st.sidebar.slider("Similarity Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.01)
run_camera = st.sidebar.checkbox("Turn On Camera Feed")

# registering new identity
st.sidebar.markdown("---")
st.sidebar.subheader("Register New Identity")
reg_name = st.sidebar.text_input("Enter Full Name", placeholder="e.g., John Cena")

if st.sidebar.button("Capture & Save Face"):
    if not run_camera:
        st.sidebar.error("Camera must be turned on to scan facial structure.")
    elif not reg_name.strip():
        st.sidebar.error("Please provide a valid name string.")
    else:
        st.session_state.register_flag = True
        st.sidebar.info("Scanning feed for a valid face layout...")

# dynamic gallery innovation
if st.sidebar.button("Adapt Profile (Rescan)"):
    if not run_camera:
        st.sidebar.error("Camera must be active to perform profile updates.")
    else:
        st.session_state.adapt_flag = True

REGISTERED_DIR = "../registered_people"
img_transform, face_cascade = load_base_assets()

with st.spinner(f"Swapping engine context to {selected_engine}..."):
    model, device, emb_dimension = load_selected_model(selected_engine)
    known_faces = precompute_database_embeddings(selected_engine, REGISTERED_DIR)

    emotion_system = load_emotion_system(
        model_path="../model/models/emotion_resnet34_transfer.pth",
        confidence_threshold=0.45,
        smoothing_window=8
    )

    antispoof_transfer = None
    try:
        antispoof_transfer = tf.keras.models.load_model("../model/models/antispoof_transfer.keras")
    except Exception as e:
        print("Anti-spoofing model not found:", e)

st.info(f"Core Active: {selected_engine} | Vector Space: {emb_dimension}D")
frame_placeholder = st.empty()

st.sidebar.subheader("Emotion Output")
current_emotion_placeholder = st.sidebar.empty()
emotion_probs_placeholder = st.sidebar.empty()

st.sidebar.subheader("Emotion Session Summary")
emotion_summary_placeholder = st.sidebar.empty()

if st.sidebar.button("Reset Emotion Summary"):
    emotion_system.reset_summary()
    st.sidebar.success("Emotion summary reset")

# main processing loop
if run_camera:
    cap = cv2.VideoCapture(0) 
    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret: break
            
        raw_frame = frame.copy()
            
        current_emb, bbox = extract_embedding(model, frame, face_cascade, img_transform, device, selected_engine)
        
        if current_emb is not None and bbox is not None:
            x, y, w, h = bbox
            
            # * IDENTITY REGISTRATION
            if st.session_state.register_flag:
                os.makedirs(REGISTERED_DIR, exist_ok=True)
                # Sanitize text characters
                clean_name = "".join([c for c in reg_name if c.isalnum() or c in (' ', '_', '-')]).strip()
                
                # UPGRADE 1: Isolate enrollment into a dedicated subfolder
                person_folder = os.path.join(REGISTERED_DIR, clean_name)
                os.makedirs(person_folder, exist_ok=True)
                file_target_path = os.path.join(person_folder, f"reg_{int(time.time())}.jpg")
                
                # write the clean frame directly to your target directory
                cv2.imwrite(file_target_path, raw_frame)
                st.session_state.register_flag = False
                
                # clear cached data to bind new target to current embedding vector arrays
                st.cache_data.clear()
                st.toast(f"Successfully registered {clean_name} into system space!")
                st.rerun()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
            
            identity = "Unknown"
            highest_score = 0.0
            
            for name, ref_emb in known_faces.items():
                score = F.cosine_similarity(current_emb, ref_emb).item()
                if score > highest_score:
                    highest_score = score
                    if score >= threshold:
                        identity = name

            emotion, emotion_confidence, avg_probs = predict_emotion(emotion_system, frame, bbox)

            if antispoof_transfer is not None:
                spoof_label, spoof_score = predict_liveness(antispoof_transfer, frame, threshold=0.55)
            else:
                spoof_label, spoof_score = "Unavailable", 0.0
            
            identity_label = f"{identity} ({highest_score:.2f})"
            liveness_label = f"Liveness: {spoof_label} ({spoof_score:.2f})"
            emotion_label = f"Emotion: {emotion} ({emotion_confidence:.2f})"

            identity_color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            liveness_color = (0, 255, 0) if spoof_label == "Real" else (0, 0, 255)
            emotion_color = (0, 255, 255) if emotion == "uncertain" else (255, 255, 0)

            cv2.putText(frame, liveness_label, (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, liveness_color, 2)
            cv2.putText(frame, identity_label, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, identity_color, 2)
            cv2.putText(frame, emotion_label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)

            # rescanning module
            if st.session_state.adapt_flag:
                st.session_state.adapt_flag = False # clear flag
                
                if identity == "Unknown":
                    st.toast("Rescan failed: No recognised identity on screen!")
                elif spoof_label != "Real":
                    st.toast(f"Security Alert: Blocked adaptation for an unverified/spoofed frame ({spoof_label})")
                else:
                    # quality-gated motion blur check (Laplacian Variance method)
                    face_crop = raw_frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                        
                        if blur_score < 70.0:
                            st.toast(f"Rescan denied: Image too blurry ({blur_score:.1f}). Please hold still.")
                        else:
                            # store multiple snapshots inside the identity folder
                            person_folder = os.path.join(REGISTERED_DIR, identity)
                            os.makedirs(person_folder, exist_ok=True)
                            
                            file_target_path = os.path.join(person_folder, f"adapt_{int(time.time())}.jpg")
                            # write the clean frame directly to your target directory
                            cv2.imwrite(file_target_path, raw_frame)
                            
                            # enforce a max 5-file rolling gallery limitation to save storage
                            history_files = sorted(
                                [os.path.join(person_folder, f) for f in os.listdir(person_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                                key=os.path.getmtime
                            )
                            while len(history_files) > 5:
                                os.remove(history_files.pop(0))
                            
                            # evict stale embeddings cache so Streamlit parses the new image look
                            st.cache_data.clear()
                            st.toast(f"Appearance updated for {identity}! Re-indexing database...")
                            st.rerun()
    
        if current_emb is not None and bbox is not None:
            if avg_probs is not None:
                emotion_prob_rows = []
                for i, emotion_name in emotion_system.emotion_classes.items():
                    emotion_prob_rows.append({
                        "Emotion": emotion_name,
                        "Probability": round(float(avg_probs[i]), 3)
                    })

                emotion_probs_placeholder.dataframe(
                    pd.DataFrame(emotion_prob_rows),
                    hide_index=True,
                    use_container_width=True
                )

        emotion_summary_placeholder.write(emotion_system.get_summary())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
    cap.release()
else:
    st.write("Toggle camera activation in the control dashboard to begin processing.")