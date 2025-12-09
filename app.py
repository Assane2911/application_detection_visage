import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# ---- CONFIGURATION GÉNÉRALE ----
st.set_page_config(
    page_title="Détection de Visages",
    page_icon="👤",
    layout="wide"
)

# ---- STYLES CSS PERSONNALISÉS ----
st.markdown("""
<style>
/* Titres */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

/* Carte moderne */
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.10);
    margin-bottom: 25px;
}

/* Sidebar design */
[data-testid="stSidebar"] {
    background-color: #f7f7f9;
}

/* Boutons modernes */
.stDownloadButton, .stButton>button {
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---- TITRE ----
st.markdown("<h1>👤 Détection de Visages — Viola-Jones</h1>", unsafe_allow_html=True)
st.write("Utilisez les paramètres à gauche pour détecter les visages dans vos images.")

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Paramètres")

uploaded_file = st.sidebar.file_uploader("📤 Importer une Image", type=["jpg", "jpeg", "png"])

scaleFactor = st.sidebar.slider(
    "Scale Factor",
    1.01, 2.0, 1.1,
    help="Plus proche de 1 → détection plus précise mais plus lente."
)

minNeighbors = st.sidebar.slider(
    "Min Neighbors",
    1, 15, 5,
    help="Augmentez pour réduire les fausses détections."
)

color_hex = st.sidebar.color_picker("🎨 Couleur du Rectangle", "#00FF00")


# ---- FONCTIONS ----
def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


@st.cache_data(show_spinner=False)
def detect_faces(image_data, cascade, scale_factor, min_n, rect_color):
    img = np.array(image_data.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, scale_factor, min_n, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), rect_color, 2)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), len(faces)


# ---- CHARGEMENT DU CLASSIFIEUR ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ---- AFFICHAGE PRINCIPAL ----
if uploaded_file:

    image_pil = Image.open(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖼️ Image Originale")
    st.image(image_pil, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("Analyse en cours... ⏳"):
        rect_color = hex_to_bgr(color_hex)
        result_img, count = detect_faces(image_pil, face_cascade, scaleFactor, minNeighbors, rect_color)

    # ---- CARTE RÉSULTAT ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✨ Résultat")

    if count > 0:
        st.success(f"🎉 {count} visage(s) détecté(s) !")
    else:
        st.warning("Aucun visage détecté. Essayez d’ajuster les paramètres.")

    st.image(result_img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- TÉLÉCHARGEMENT ----
    buf = BytesIO()
    Image.fromarray(result_img).save(buf, format="PNG")

    st.download_button(
        "⬇️ Télécharger l'image détectée",
        data=buf.getvalue(),
        file_name="visages_detectes.png",
        mime="image/png"
    )

else:
    st.info("➕ Importez une image dans la barre latérale pour commencer.")
