import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import gdown

# Streamlit page setup
st.set_page_config(page_title="Moringa Disease Detector")
st.title("🌿 Moringa Disease Detector")

# Model file info
MODEL_PATH = "Moringa2Classifier.pth"
FILE_ID = "1rTFiBXvKznKNqW9Lg-wMYi0HsWK68vfx"

# Load the model, downloading it if necessary
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?export=download&id={FILE_ID}", MODEL_PATH, quiet=False)
    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()
    return model

model = load_cnn_model()

# Moringa class names
class_names = [
    "Bacterial Leaf Spot", 
    "Cercospora Leaf Spot", 
    "Healthy Leaf", 
    "Yellow Leaf"
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Preprocessing
def preprocess(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# Image upload UI
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
