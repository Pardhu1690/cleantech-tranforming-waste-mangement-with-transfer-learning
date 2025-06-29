
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('models/waste_classifier_model.h5')
labels = ['Glass', 'Hazardous', 'Metal', 'Organic', 'Plastic', 'Recyclable']

st.title("♻️ CleanTech: Waste Classification App")
st.markdown("Upload a waste image to classify it into one of the categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        index = np.argmax(predictions)
        confidence = round(100 * np.max(predictions), 2)

        st.success(f"Prediction: **{labels[index]}** with {confidence}% confidence")
