import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import os

st.set_page_config(page_title="Vegetable Classifier", layout="centered")

st.title("🥦 Vegetable Image Classifier")
st.markdown("Upload a vegetable image and let the model predict its class.")

# Load model from Google Drive link
@st.cache_resource
def load_keras_model():
    # If model file not available, download from Google Drive
    if not os.path.exists("vegetable_model.h5"):
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
        r = requests.get(url, allow_redirects=True)
        open('vegetable_model.h5', 'wb').write(r.content)

    return load_model("vegetable_model.h5")

model = load_keras_model()

# Define your class names (change these to match your model)
class_names = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]

uploaded_file = st.file_uploader("Choose a vegetable image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")
