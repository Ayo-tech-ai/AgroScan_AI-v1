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
st.write("Upload an image of a moringa leaf and get a prediction of its health condition.")

# -------------------------------
# 📦 Model Download & Load
# -------------------------------
MODEL_PATH = "Moringa2Classifier.pth"
FILE_ID = "1rTFiBXvKznKNqW9Lg-wMYi0HsWK68vfx"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# 🏷️ Class Labels
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
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# -------------------------------
# 📤 Image Upload & Prediction
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                input_tensor = preprocess_image(image)
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)

                label = class_names[predicted_class.item()]
                confidence_score = confidence.item() * 100

                st.success(f"**Prediction:** {label}")
                st.info(f"**Confidence Level:** {confidence_score:.2f}%")
    except Exception as e:
        st.error("❌ Failed to process image. Please upload a valid image file.")
        st.exception(e)
