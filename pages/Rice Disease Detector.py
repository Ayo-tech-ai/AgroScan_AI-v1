import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import gdown

# --------------------------------------
# 🧠 Page Config and Title
# --------------------------------------
st.set_page_config(page_title="🌾 Rice Disease Detector")
st.title("🌾 Rice Disease Detector")

# --------------------------------------
# 📦 Model Loading
# --------------------------------------
MODEL_PATH = "RiceClassifier.pth"
FILE_ID = "13nlieOIczZPmbCaA8M2AlefOrXINTXyL"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()
    return model

model = load_model()

# --------------------------------------
# 🏷️ Class Names
# --------------------------------------
class_names = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bakanae',
    'brown_spot', 'grassy_stunt_virus', 'healthy_rice_plant',
    'narrow_brown_spot', 'ragged_stunt_virus', 'rice_blast',
    'rice_false_smut', 'sheath_blight', 'sheath_rot',
    'stem_rot', 'tungro_virus'
]

# --------------------------------------
# 🔄 Image Preprocessing
# --------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_image(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# --------------------------------------
# 📤 Upload & Predict
# --------------------------------------
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")  # Removed use_container_width=True

    if st.button("Classify Disease"):
        with st.spinner("Analyzing image..."):
            try:
                input_tensor = preprocess_image(image)
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probs, 1)

                label = class_names[prediction.item()]
                score = confidence.item() * 100

                st.success(f"🌿 Prediction: **{label}**")
                st.info(f"📊 Confidence: **{score:.2f}%**")

            except Exception as e:
                st.error("⚠️ Something went wrong during classification.")
                st.exception(e)
