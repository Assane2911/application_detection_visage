import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Détection de Visages", page_icon="👤", layout="wide")

st.title("👤 Détection de Visages avec Viola-Jones")
st.caption("Application utilisant OpenCV pour détecter les visages.")

# Charger le classifieur
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Upload
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Détection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Dessiner les rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

    st.image(img_array, caption="Résultat de la détection", use_column_width=True)
else:
    st.info("Veuillez télécharger une image pour commencer.")
