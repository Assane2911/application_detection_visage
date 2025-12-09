import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Détection de Visages avec Viola-Jones",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Détection de Visages avec Viola-Jones (Streamlit)")
st.caption("Application utilisant OpenCV pour détecter les visages dans une image téléchargée.")

# --- 1. Instructions pour l'Utilisateur ---
st.markdown("""
## 📝 Instructions d'Utilisation
1.  **Téléchargez** une image contenant des visages dans la barre latérale (**Upload Image**).
2.  Ajustez les **Paramètres de Détection** (`Scale Factor` et `Min Neighbors`) dans la barre latérale pour optimiser la détection.
3.  Choisissez la **Couleur du Rectangle** pour les cadres de détection.
4.  L'image traitée s'affichera ci-dessous.
5.  Cliquez sur le bouton **Télécharger l'image** pour sauvegarder le résultat sur votre appareil.
""")
st.markdown("---")

# --- Initialisation du Classifieur Cascade de Visages ---
# Nous supposons que le fichier 'haarcascade_frontalface_default.xml' est disponible localement.
try:
    # Charger le classifieur (chemin à ajuster si nécessaire)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Erreur: Le fichier 'haarcascade_frontalface_default.xml' n'a pas pu être chargé. Assurez-vous qu'il est disponible dans le chemin d'accès d'OpenCV.")
except Exception as e:
    st.error(f"Erreur lors du chargement du classifieur: {e}")
    face_cascade = None

# --- Barre Latérale pour les Contrôles (Paramètres) ---
st.sidebar.header("⚙️ Paramètres & Contrôles")

# Contrôle du téléchargement d'image
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# --- 5. Ajustement de scaleFactor ---
# Le scaleFactor doit être > 1.0. Valeur recommandée : 1.05 - 1.4.
scaleFactor = st.sidebar.slider(
    "Scale Factor (`scaleFactor`)",
    min_value=1.01,
    max_value=2.0,
    value=1.1,
    step=0.01,
    help="Facteur de réduction de l'image pour l'étape de détection. Une valeur plus petite (proche de 1.01) augmente la précision mais ralentit le traitement."
)

# --- 4. Ajustement de minNeighbors ---
# minNeighbors : nombre de voisins qu'un candidat rectangle doit avoir pour être conservé.
minNeighbors = st.sidebar.slider(
    "Min Neighbors (`minNeighbors`)",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="Nombre minimal de voisins (détections) qu'un rectangle candidat doit posséder pour être considéré comme un visage. Une valeur plus élevée réduit les fausses détections."
)

# --- 3. Sélecteur de Couleur ---
# st.color_picker() retourne une couleur en format hexadécimal (ex: #FF0000)
color_hex = st.sidebar.color_picker(
    'Couleur du Rectangle de Détection',
    '#00FF00',  # Vert par défaut
    help="Choisissez la couleur du cadre dessiné autour des visages détectés."
)

# Fonction pour convertir HEX en RGB (pour OpenCV)
def hex_to_bgr(hex_color):
    # Convertir HEX (#RRGGBB) en tuple BGR (Blue, Green, Red)
    hex_color = hex_color.lstrip('#')
    # Les tuples de cv2 sont (B, G, R)
    b = int(hex_color[4:6], 16)
    g = int(hex_color[2:4], 16)
    r = int(hex_color[0:2], 16)
    return (b, g, r)

rectangle_color_bgr = hex_to_bgr(color_hex)
rectangle_thickness = 2 # Épaisseur du rectangle

# --- Fonction Principale de Détection ---
@st.cache_data(show_spinner=False)
def detect_faces(image_data, cascade, scale_factor, min_n):
    """
    Détecte les visages dans une image en utilisant l'algorithme de Viola-Jones.
    """
    # Convertir l'image PIL/Streamlit en array numpy pour OpenCV
    img = np.array(image_data.convert('RGB'))
    # Convertir RGB en BGR pour OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convertir en niveaux de gris pour la détection (c'est plus rapide)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Lancement de la détection de visages
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_n,
        minSize=(30, 30) # Taille minimale du visage à détecter (optionnel)
    )

    # Dessiner les rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), rectangle_color_bgr, rectangle_thickness)

    # Reconvertir BGR en RGB pour l'affichage Streamlit (qui utilise RGB)
    result_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return result_img_rgb, len(faces)

# --- Affichage et Traitement ---
if uploaded_file is not None and face_cascade is not None:
    try:
        # Lire l'image téléchargée
        image_pil = Image.open(uploaded_file)
        
        # Afficher l'image originale
        st.header("🖼️ Image Originale")
        st.image(image_pil, caption=f"Image : {uploaded_file.name}", use_column_width=True)
        st.markdown("---")
        
        # Traiter l'image
        st.subheader("⏳ Traitement en cours...")
        
        result_img_rgb, face_count = detect_faces(image_pil, face_cascade, scaleFactor, minNeighbors)
        
        st.header("✨ Résultat de la Détection")
        
        if face_count > 0:
            st.success(f"✅ {face_count} visage(s) détecté(s)!")
        else:
            st.warning("⚠️ Aucun visage détecté. Essayez d'ajuster les paramètres dans la barre latérale.")

        # Afficher l'image résultante
        st.image(result_img_rgb, caption="Visages détectés", use_column_width=True)

        # --- 2. Fonction de Sauvegarde (Téléchargement) ---
        # Convertir l'array numpy RGB résultant en image PIL, puis en bytes pour le téléchargement
        
        # OpenCV/Numpy est en RGB, Streamlit le gère correctement
        result_image_pil = Image.fromarray(result_img_rgb)
        
        # Créer un objet BytesIO pour le téléchargement
        from io import BytesIO
        buf = BytesIO()
        result_image_pil.save(buf, format="PNG") # Utiliser PNG pour une meilleure qualité
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Télécharger l'image des visages détectés (PNG)",
            data=byte_im,
            file_name=f"visages_detectes_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement de l'image: {e}")

elif uploaded_file is None:
    st.info("Veuillez télécharger une image pour commencer la détection de visages.")

elif face_cascade is None:
    st.error("L'application ne peut pas fonctionner car le classifieur de visages n'a pas pu être chargé. Veuillez vérifier l'installation d'OpenCV.")
