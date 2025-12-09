import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Détection de visage", page_icon="👤")

st.title("Détection de visage")

# Toujours les widgets en haut
uploaded_file = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])

# Charger une fois
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Image originale")

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Dessin
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

    st.image(img_array, caption="Image avec détection")
else:
    st.info("Veuillez importer une image.")
