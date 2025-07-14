import streamlit as st
import joblib

st.set_page_config(page_title="Crop Recommender")
st.title("🌾 Crop Recommendation Tool")

@st.cache_resource
def load_rf_model():
    return joblib.load("CropRec.joblib"), joblib.load("CropRec_LabelEncoder.joblib")

model, encoder = load_rf_model()

N = st.number_input("Nitrogen (N)", 0, 200)
P = st.number_input("Phosphorus (P)", 0, 200)
K = st.number_input("Potassium (K)", 0, 200)
temp = st.number_input("Temperature (°C)", 10.0, 45.0)
humidity = st.number_input("Humidity (%)", 10.0, 100.0)
ph = st.number_input("pH", 3.0, 9.0)
rainfall = st.number_input("Rainfall (mm)", 10.0, 2000.0)

if st.button("Recommend Crop"):
    features = [[N, P, K, temp, humidity, ph, rainfall]]
    pred = model.predict(features)
    crop = encoder.inverse_transform(pred)
    st.success(f"Recommended Crop: {crop[0]}")
