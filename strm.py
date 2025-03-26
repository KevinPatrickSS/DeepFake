import streamlit as st
import cv2
import torch
from PIL import Image
import timm
import numpy as np
import torchvision.transforms as transforms
import tempfile
import os

# ✅ Set Device (Use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load Pretrained Model
st.title("🎭 Real-Time DeepFake Detection")
st.sidebar.header("Upload a Video for Analysis")

@st.cache_resource
def load_model():
    model = timm.create_model("xception", pretrained=True)
    model.fc = torch.nn.Linear(model.num_features, 1)  
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# ✅ Define Transformation
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])

# ✅ DeepFake Video Analysis Function
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fake_scores = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)  # Streamlit progress bar
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        frame_tensor = transform(frame_pil)
        frame_tensor = frame_tensor.unsqueeze(0).to(device) 

        with torch.no_grad():
            output = model(frame_tensor)
            pred = torch.sigmoid(output.squeeze()).item()  
            fake_scores.append(pred)

    cap.release()
    
    # ✅ Compute Final Video Classification
    avg_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0
    final_decision = "Fake" if avg_score > 0.5 else "Real"

    return final_decision, avg_score

# ✅ Streamlit File Uploader
uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save video to temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.video(temp_video_path)  # Display video

    if st.sidebar.button("Analyze Video"):
        with st.spinner("Analyzing video... Please wait!"):
            result, score = analyze_video(temp_video_path)
            st.success(f"🎥 Final Classification: **{result}** (Score: {score:.4f})")
    
    os.remove(temp_video_path)  # Clean up temporary file
