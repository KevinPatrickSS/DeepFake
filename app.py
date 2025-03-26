import streamlit as st
import cv2
import torch
import timm
import numpy as np
from PIL import Image
import tempfile
import torchvision.transforms as transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
@st.cache(allow_output_mutation=True)
def load_model():
    model = timm.create_model("xception", pretrained=True)
    model.fc = torch.nn.Linear(model.num_features, 1)  # Adjust for binary classification
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fake_scores = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return "Error: Cannot read video", 0.0 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        frame_tensor = transform(frame_pil)
        frame_tensor = frame_tensor.unsqueeze(0).to(device) 

        with torch.no_grad():
            output = model(frame_tensor)
            pred = torch.sigmoid(output.squeeze()).item() 
            fake_scores.append(pred)

    cap.release()

    avg_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0
    final_decision = "Fake" if avg_score > 0.5 else "Real"
    
    return final_decision, avg_score

st.title("🔍 DeepFake Video Detector")
st.write("Upload a video and this tool will analyze if it's **Real or Fake**.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name

    st.video(temp_file_path)
    if st.button("Analyze Video"):
        with st.spinner("Processing... Please wait."):
            result, avg_score = analyze_video(temp_file_path)
        
        st.success(f"🎥 **Final Classification: {result}**")
        st.write(f"**Confidence Score:** {avg_score:.4f}")
