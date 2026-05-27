import cv2
import torch
import torch.nn as nn
from PIL import Image
from collections import deque, Counter
from torchvision import models, transforms


class EmotionInnovation:
    def __init__(
        self,
        model_path="../models/emotion_resnet34_transfer.pth",
        confidence_threshold=0.45,
        smoothing_window=8,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emotion_classes = {
            0: "surprise",
            1: "fear",
            2: "disgust",
            3: "happy",
            4: "sad",
            5: "angry",
            6: "neutral"
        }

        self.confidence_threshold = confidence_threshold
        self.probability_history = deque(maxlen=smoothing_window)
        self.session_counts = Counter()

        self.model = models.resnet34(weights=None)

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, len(self.emotion_classes))
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def crop_face(self, frame, bbox):
        x, y, w, h = bbox

        # Tighter crop for emotion detection.
        # This reduces background/head area and keeps the model focused on the face.
        padding_x = int(w * 0.08)
        padding_y_top = int(h * 0.05)
        padding_y_bottom = int(h * 0.08)

        x1 = max(x + padding_x, 0)
        y1 = max(y + padding_y_top, 0)
        x2 = min(x + w - padding_x, frame.shape[1])
        y2 = min(y + h - padding_y_bottom, frame.shape[0])

        face_crop = frame[y1:y2, x1:x2]

        return face_crop

    def predict(self, frame, bbox):
        face_crop = self.crop_face(frame, bbox)

        if face_crop.size == 0:
            return "unknown", 0.0, None

        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)

        input_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu()

        # Instead of trusting a single frame, I keep recent probabilities.
        self.probability_history.append(probabilities)

        # Average recent probabilities to reduce flickering between frames.
        avg_probs = torch.stack(list(self.probability_history)).mean(dim=0)

        confidence, predicted_class = torch.max(avg_probs, dim=0)

        confidence_score = float(confidence)
        predicted_index = int(predicted_class)
        predicted_emotion = self.emotion_classes[predicted_index]

        # If confidence is low, don't force an emotion label.
        if confidence_score < self.confidence_threshold:
            final_emotion = "uncertain"
        else:
            final_emotion = predicted_emotion
            self.session_counts[final_emotion] += 1

        return final_emotion, confidence_score, avg_probs

    def draw_probability_bars(self, frame, avg_probs, start_x=20, start_y=90):
        if avg_probs is None:
            return frame

        bar_width = 150
        bar_height = 16
        gap = 8

        for i in range(len(self.emotion_classes)):
            emotion = self.emotion_classes[i]
            prob = float(avg_probs[i])

            y = start_y + i * (bar_height + gap)

            cv2.putText(
                frame,
                f"{emotion}: {prob:.2f}",
                (start_x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1
            )

            cv2.rectangle(
                frame,
                (start_x, y),
                (start_x + bar_width, y + bar_height),
                (80, 80, 80),
                1
            )

            cv2.rectangle(
                frame,
                (start_x, y),
                (start_x + int(bar_width * prob), y + bar_height),
                (0, 255, 255),
                -1
            )

        return frame

    def reset_summary(self):
        self.probability_history.clear()
        self.session_counts.clear()

    def get_summary(self):
        return dict(self.session_counts)


def load_emotion_system(
    model_path="../models/emotion_resnet34_transfer.pth",
    confidence_threshold=0.45,
    smoothing_window=8
):
    return EmotionInnovation(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        smoothing_window=smoothing_window
    )


def predict_emotion(emotion_system, frame, bbox):
    return emotion_system.predict(frame, bbox)