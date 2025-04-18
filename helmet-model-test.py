import torch
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
import threading

# --- Non-blocking alert function ---
def show_alert_non_blocking():
    def run_alert():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Warning!", "User has been looking for more than 5 seconds!")
        root.destroy()
    threading.Thread(target=run_alert).start()

# --- Timer Variables ---
looking_start_time = None
last_alert_time = 0
LOOKING_THRESHOLD = 5   # seconds to first alert  
ALERT_INTERVAL = 15    # seconds between alerts if still looking

# --- Speed Functionality ---
SPEED_THRESHOLD_KMPH = 25
current_speed_kmph = 26  # Simulated Speed - Change this dynamically for testing.

# --- SETUP ---
classes = ["looking", "not_looking"]
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# --- LOAD MODEL ---
model = efficientnet_b3(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("efficientnet_b3_final.pth", map_location=device))
model.to(device)
model.eval()

# --- TRANSFORM (match training) ---
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])         

# --- OPENCV VIDEO LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB and apply transform
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        label = classes[predicted.item()]
        confidence = conf.item()

    current_time = time.time()

    # --- Alert logic ---
    if label == "looking":
        if looking_start_time is None:
            looking_start_time = current_time
        elif (current_time - looking_start_time >= LOOKING_THRESHOLD and 
            current_time - last_alert_time >= ALERT_INTERVAL and 
            current_speed_kmph >= SPEED_THRESHOLD_KMPH):
            show_alert_non_blocking()
            last_alert_time = current_time
    else:
        looking_start_time = None
        last_alert_time = 0

    # --- TEXT DISPLAY CONFIG ---
    color = (0, 255, 0) if label == "not_looking" else (0, 0, 255)
    text = f"{label.upper()} ({confidence * 100:.1f}%)"

    # Draw text
    cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # Display
    cv2.imshow("Helmet Pose Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
