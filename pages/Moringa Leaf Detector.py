import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Moringa Disease Detector")
st.title("🌿 Moringa Disease Detector")

MODEL_PATH = "Moringa2Classifier.pth"
FILE_ID = "YOUR_FILE_ID_HERE"  # ⬅️ Replace this with the real ID

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    return model

model = load_cnn_model()

class_names = [
    "Bacterial Leaf Spot", 
    "Cercospora Leaf Spot", 
    "Healthy Leaf", 
    "Yellow Leaf"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

img = st.file_uploader("Upload a moringa leaf image", type=["jpg", "jpeg", "png"])
if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Disease"):
        tensor = preprocess(image)
        with st.spinner("Classifying..."):
            out = model(tensor)
            probs = F.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
            st.success(f"Prediction: {class_names[pred.item()]} ({conf.item() * 100:.2f}%)")
