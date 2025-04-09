import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_PATH = "vegetable_classifier_v02.keras"
FILE_ID = "1pRNf5zErEnb0n51B0SBS42ng96kzR1VV"  # Replace with your actual file ID

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Load the model (.keras format)
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Beetroot', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage',
               'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato',
               'Pumpkin', 'Radish', 'Tomato']

st.title("🥦 Vegetable Classifier 🍅")
st.markdown("Upload a vegetable image and get its prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index]

    st.success(f"**Prediction:** {predicted_label} ({confidence:.2f} confidence)")
