import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import gdown

# -------------------------------
# 🔧 Page Configuration
# -------------------------------
st.set_page_config(page_title="Moringa Disease Detector", layout="centered")
st.title("🌿 Moringa Disease Detector")

# -------------------------------
# 📦 Model Download & Load
# -------------------------------
MODEL_PATH = "Moringa2Classifier.pth"
FILE_ID = "1rTFiBXvKznKNqW9Lg-wMYi0HsWK68vfx"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?export=download&id={FILE_ID}", MODEL_PATH, quiet=False)
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_cnn_model()

# -------------------------------
# 🏷️ Class Names
# -------------------------------
class_names = [
    "Bacterial Leaf Spot", 
    "Cercospora Leaf Spot", 
    "Healthy Leaf", 
    "Yellow Leaf"
]

# -------------------------------
# 🔁 Image Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# -------------------------------
# 📤 Image Upload & Inference
# -------------------------------
uploaded_file = st.file_uploader("Upload a Moringa leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Disease"):
            with st.spinner("Classifying..."):
                input_tensor = preprocess(image)
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
                st.success(f"🩺 Prediction: **{class_names[pred.item()]}** ({conf.item() * 100:.2f}% confidence)")
    except Exception as e:
        st.error("❌ Failed to process image. Please upload a valid image file.")
        st.exception(e)
