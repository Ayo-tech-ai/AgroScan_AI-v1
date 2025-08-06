import streamlit as st

st.set_page_config(page_title="AgroScan AI", layout="centered")

st.title("🌱 Welcome to AgroScan AI")
st.markdown("""
AgroScan AI is your unified, AI-powered assistant for smart agriculture.
Choose one of the tools from the sidebar to:

- Detect **Rice leaf diseases** using image classification  
- Detect **Moringa leaf diseases** with visual explanations  
- Get **Crop Recommendations** based on soil and climate data  
""")
