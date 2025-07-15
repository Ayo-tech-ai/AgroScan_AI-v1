import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tf_explain.core.grad_cam import GradCAM

st.set_page_config(page_title="Moringa Detector")
st.title("🌿 Moringa Leaf Detector")

@st.cache_resource
def load_model():
    m = tf.keras.models.load_model("moringa_effnet_final.keras")
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return m

model = load_model()
backbone = model.get_layer("efficientnetb0")
class_names = ["Bacterial Leaf Spot", "Cercospora Leaf Spot", "Healthy Leaf", "Yellow Leaf"]

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(np.array(img, dtype="float32"), axis=0))
    return arr

img = st.file_uploader("Upload a moringa leaf image", type=["jpg", "jpeg", "png"])
if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        arr = preprocess(image)
        preds = model(arr, training=False)
        class_idx = int(np.argmax(preds))
        st.success(f"Predicted Class: {class_names[class_idx]}")
      
