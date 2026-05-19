import streamlit as st
import tensorflow as tf
import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import time

# =========================================================
# PAGE CONFIG — must be first Streamlit call
# =========================================================

st.set_page_config(
    page_title="ForestWatch · Détection IA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }

  /* ── Deep forest background ── */
  .stApp {
    background: #040d08;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(20,90,40,0.35) 0%, transparent 70%),
      url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%231a3d25' fill-opacity='0.08'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
  }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0a1f0f 0%, #0d2b15 40%, #071409 100%);
    border: 1px solid rgba(52, 199, 89, 0.18);
    border-radius: 20px;
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(52,199,89,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 20%;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(52,199,89,0.07) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-tag {
    display: inline-block;
    background: rgba(52,199,89,0.12);
    color: #34c759;
    border: 1px solid rgba(52,199,89,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }
  .hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #f0faf3;
    line-height: 1.1;
    margin-bottom: 0.75rem;
  }
  .hero-title span {
    color: #34c759;
  }
  .hero-subtitle {
    font-size: 1rem;
    color: #6b8f73;
    max-width: 520px;
    line-height: 1.6;
    font-weight: 400;
  }
  .hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(52,199,89,0.1);
  }
  .stat-item { display: flex; flex-direction: column; gap: 2px; }
  .stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #34c759;
  }
  .stat-label {
    font-size: 0.72rem;
    color: #4a6652;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Upload zone ── */
  .upload-zone {
    background: #071409;
    border: 1.5px dashed rgba(52,199,89,0.25);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
    margin-bottom: 1.5rem;
  }
  .upload-zone:hover { border-color: rgba(52,199,89,0.5); }
  .upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
  .upload-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #c8e6cb;
    margin-bottom: 0.3rem;
  }
  .upload-hint {
    font-size: 0.8rem;
    color: #4a6652;
    font-family: 'Space Mono', monospace;
  }

  /* ── Section headers ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(52,199,89,0.1);
  }
  .section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #34c759;
    box-shadow: 0 0 8px rgba(52,199,89,0.6);
  }
  .section-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #34c759;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Space Mono', monospace;
  }

  /* ── Info card ── */
  .info-card {
    background: #0a1a0e;
    border: 1px solid rgba(52,199,89,0.12);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
  }
  .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(52,199,89,0.06);
  }
  .info-row:last-child { border-bottom: none; }
  .info-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4a6652;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .info-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #a8d5b0;
    font-weight: 700;
  }

  /* ── Result card ── */
  .result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .result-card.forest {
    background: linear-gradient(135deg, #052010 0%, #0a2e15 100%);
    border: 1.5px solid rgba(52,199,89,0.35);
    box-shadow: 0 0 40px rgba(52,199,89,0.08), inset 0 1px 0 rgba(52,199,89,0.15);
  }
  .result-card.non-forest {
    background: linear-gradient(135deg, #1a0a05 0%, #2a1205 100%);
    border: 1.5px solid rgba(255,149,0,0.35);
    box-shadow: 0 0 40px rgba(255,149,0,0.08), inset 0 1px 0 rgba(255,149,0,0.15);
  }
  .result-label {
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
  }
  .result-label.forest { color: #34c759; }
  .result-label.non-forest { color: #ff9500; }
  .result-class {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.25rem;
  }
  .result-class.forest { color: #f0faf3; }
  .result-class.non-forest { color: #fff7f0; }
  .result-proba {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin: 1rem 0;
  }
  .result-proba.forest { color: #34c759; }
  .result-proba.non-forest { color: #ff9500; }
  .result-desc {
    font-size: 0.82rem;
    color: #6b8f73;
    line-height: 1.5;
  }

  /* ── Gauge bar ── */
  .gauge-wrap {
    background: #071409;
    border: 1px solid rgba(52,199,89,0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
  }
  .gauge-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #4a6652;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
  }
  .gauge-track {
    height: 8px;
    background: #0d1f10;
    border-radius: 4px;
    overflow: hidden;
  }
  .gauge-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
  }
  .gauge-fill.forest {
    background: linear-gradient(90deg, #1a7a30, #34c759);
    box-shadow: 0 0 10px rgba(52,199,89,0.4);
  }
  .gauge-fill.non-forest {
    background: linear-gradient(90deg, #b35800, #ff9500);
    box-shadow: 0 0 10px rgba(255,149,0,0.4);
  }
  .gauge-ticks {
    display: flex;
    justify-content: space-between;
    margin-top: 0.4rem;
  }
  .gauge-tick {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #2a4030;
  }
  .gauge-tick.active { color: #34c759; }

  /* ── Alert boxes ── */
  .stAlert > div {
    background: #0a1a0e !important;
    border-color: rgba(52,199,89,0.25) !important;
    color: #a8d5b0 !important;
    border-radius: 10px !important;
  }

  /* ── Streamlit file uploader ── */
  [data-testid="stFileUploader"] {
    background: transparent !important;
  }
  [data-testid="stFileUploader"] > div {
    background: #071409 !important;
    border: 1.5px dashed rgba(52,199,89,0.25) !important;
    border-radius: 16px !important;
  }
  [data-testid="stFileUploader"] label {
    color: #a8d5b0 !important;
  }
  [data-testid="stFileUploadDropzone"] {
    background: transparent !important;
  }

  /* ── Image display ── */
  [data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(52,199,89,0.15);
  }

  /* ── Streamlit default text overrides ── */
  p, li, label { color: #a8d5b0 !important; }
  h1, h2, h3 { color: #f0faf3 !important; }

  /* ── Spinner ── */
  .stSpinner > div { color: #34c759 !important; }

  /* ── Success / error ── */
  .element-container .stSuccess {
    background: #071d0a !important;
    border-left-color: #34c759 !important;
  }
  .element-container .stError {
    background: #1a0705 !important;
    border-left-color: #ff453a !important;
  }

  /* ── Footer ── */
  .app-footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(52,199,89,0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .footer-brand {
    font-weight: 800;
    font-size: 0.9rem;
    color: #2a4030;
  }
  .footer-brand span { color: #34c759; }
  .footer-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #1e3225;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Divider ── */
  hr { border-color: rgba(52,199,89,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_DIR = "."
MODEL_PATH = os.path.join(MODEL_DIR, "meilleur_cnn.keras")
TAILLE = 64
SEUIL_FINAL = 0.64
FOCAL_ALPHA = 0.58
FOCAL_GAMMA = 2.0

# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================

def nettoyer(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


# =========================================================
# FOCAL LOSS
# =========================================================

@tf.keras.utils.register_keras_serializable(name='loss_fn')
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, name='focal_loss_obj'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, self.gamma) * bce)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config


# =========================================================
# MÉTRIQUE F1
# =========================================================

@tf.keras.utils.register_keras_serializable()
def f1_metrique(y_true, y_pred):
    y_p = tf.cast(y_pred >= 0.5, tf.float32)
    y_t = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(y_t * y_p)
    fp = tf.reduce_sum((1 - y_t) * y_p)
    fn = tf.reduce_sum(y_t * (1 - y_p))
    prec = tp / (tp + fp + 1e-7)
    rec  = tp / (tp + fn + 1e-7)
    return (2 * prec * rec) / (prec + rec + 1e-7)


# =========================================================
# CHARGEMENT IMAGE
# =========================================================

def charger_image_pour_prediction(file_bytes, taille=TAILLE):
    try:
        img = tiff.imread(io.BytesIO(file_bytes)).astype(np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[0] <= 10:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] > 3:
            img = img[:, :, :3]
        mn, mx = np.min(img), np.max(img)
        img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((taille, taille))
        return np.array(img_pil).astype(np.float32) / 255.0
    except Exception:
        try:
            img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img_pil = img_pil.resize((taille, taille))
            return np.array(img_pil).astype(np.float32) / 255.0
        except Exception as e:
            st.error(f"Erreur chargement image : {e}")
            return None


# =========================================================
# CHARGEMENT MODÈLE
# =========================================================

@st.cache_resource
def load_deforestation_model():
    custom_objects = {
        'loss_fn': FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
        'f1_metrique': f1_metrique
    }
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        return model, True
    except Exception as e:
        return None, str(e)


# =========================================================
# HERO BANNER
# =========================================================

st.markdown("""
<div class="hero-banner">
  <div class="hero-tag">🛰 Surveillance satellitaire · IA CNN</div>
  <div class="hero-title">Forest<span>Watch</span></div>
  <div class="hero-subtitle">
    Système de détection de déforestation par intelligence artificielle —
    analyse des images satellites pour identifier la couverture forestière
    en temps réel.
  </div>
  <div class="hero-stats">
    <div class="stat-item">
      <div class="stat-value">CNN</div>
      <div class="stat-label">Architecture</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">64×64</div>
      <div class="stat-label">Résolution</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">0.64</div>
      <div class="stat-label">Seuil décision</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">TIFF / JPG</div>
      <div class="stat-label">Formats acceptés</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CHARGEMENT MODÈLE
# =========================================================

cnn_model, model_status = load_deforestation_model()

if model_status is True:
    st.success("✅ Modèle chargé — prêt pour l'analyse")
else:
    st.error(f"⚠️ Modèle introuvable : {model_status}")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# =========================================================
# LAYOUT — 2 COLONNES
# =========================================================

col_left, col_right = st.columns([1, 1], gap="large")

# ── Colonne gauche : upload ──────────────────────────────
with col_left:
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Entrée image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Déposer une image satellite",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="
          background:#071409;
          border:1.5px dashed rgba(52,199,89,0.15);
          border-radius:12px;
          padding:2rem;
          text-align:center;
          margin-top:0.5rem;
        ">
          <div style="font-size:2.5rem; margin-bottom:0.5rem;">🌿</div>
          <div style="color:#4a6652; font-size:0.82rem; font-family:'Space Mono',monospace;">
            Formats supportés : TIFF · JPG · PNG
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Affichage image chargée ──────────────────────────
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem">
          <div class="section-dot"></div>
          <div class="section-title">Aperçu image</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            if uploaded_file.name.lower().endswith((".tif", ".tiff")):
                img_display = tiff.imread(io.BytesIO(file_bytes)).astype(np.float32)
                img_display = np.nan_to_num(img_display, nan=0.0, posinf=0.0, neginf=0.0)

                shape_orig = img_display.shape
                dtype_orig = img_display.dtype

                if len(img_display.shape) == 2:
                    img_display = np.stack([img_display] * 3, axis=-1)
                if img_display.shape[0] <= 10:
                    img_display = np.transpose(img_display, (1, 2, 0))
                if img_display.shape[2] > 3:
                    img_display = img_display[:, :, :3]

                mn, mx = np.min(img_display), np.max(img_display)
                img_display = (img_display - mn) / (mx - mn) if mx > mn else np.zeros_like(img_display)
                img_display = (img_display * 255).astype(np.uint8)

                st.image(img_display, use_container_width=True)

                st.markdown(f"""
                <div class="info-card">
                  <div class="info-row">
                    <span class="info-key">Fichier</span>
                    <span class="info-val">{uploaded_file.name}</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Dimensions</span>
                    <span class="info-val">{shape_orig}</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Type pixel</span>
                    <span class="info-val">{dtype_orig}</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Valeurs</span>
                    <span class="info-val">[{img_display.min()} — {img_display.max()}]</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Format</span>
                    <span class="info-val">GeoTIFF</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                w, h = img_pil.size
                st.image(img_pil, use_container_width=True)
                st.markdown(f"""
                <div class="info-card">
                  <div class="info-row">
                    <span class="info-key">Fichier</span>
                    <span class="info-val">{uploaded_file.name}</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Dimensions</span>
                    <span class="info-val">{w} × {h} px</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Format</span>
                    <span class="info-val">Raster RGB</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur affichage : {e}")


# ── Colonne droite : résultats ───────────────────────────
with col_right:
    st.markdown("""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Analyse CNN</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div style="
          background:#071409;
          border:1px solid rgba(52,199,89,0.1);
          border-radius:16px;
          padding:3rem 2rem;
          text-align:center;
          margin-top:0.5rem;
        ">
          <div style="font-size:2rem; margin-bottom:1rem; opacity:0.4;">📡</div>
          <div style="color:#2a4030; font-size:0.85rem; font-family:'Space Mono',monospace; line-height:1.7;">
            En attente d'une image<br>pour démarrer l'analyse…
          </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None and cnn_model is not None:
        with st.spinner("Analyse en cours…"):
            image_test = charger_image_pour_prediction(file_bytes, taille=TAILLE)

        if image_test is not None:
            image_batch = np.expand_dims(image_test, axis=0)
            proba = float(cnn_model.predict(image_batch, verbose=0)[0][0])
            classe = int(proba >= SEUIL_FINAL)

            is_forest   = classe == 1
            card_cls    = "forest" if is_forest else "non-forest"
            emoji       = "🌳" if is_forest else "⚠️"
            label_txt   = "FORÊT DÉTECTÉE" if is_forest else "NON-FORÊT"
            class_txt   = "Forêt" if is_forest else "Non-forêt"
            desc_txt    = (
                "La zone analysée présente une couverture végétale dense "
                "caractéristique d'un couvert forestier intact."
                if is_forest else
                "La zone analysée ne présente pas les caractéristiques "
                "spectrales d'un couvert forestier — déforestation possible."
            )
            proba_pct = f"{proba * 100:.1f}%"
            bar_pct   = int(proba * 100)

            st.markdown(f"""
            <div class="result-card {card_cls}">
              <div class="result-label {card_cls}">{emoji} {label_txt}</div>
              <div class="result-class {card_cls}">{class_txt}</div>
              <div class="result-proba {card_cls}">{proba_pct}</div>
              <div class="result-desc">{desc_txt}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge bar
            st.markdown(f"""
            <div class="gauge-wrap">
              <div class="gauge-label">
                <span>Probabilité de forêt</span>
                <span style="color:{'#34c759' if is_forest else '#ff9500'}">{proba:.4f}</span>
              </div>
              <div class="gauge-track">
                <div class="gauge-fill {card_cls}" style="width:{bar_pct}%"></div>
              </div>
              <div class="gauge-ticks">
                <span class="gauge-tick">0.0</span>
                <span class="gauge-tick">0.25</span>
                <span class="gauge-tick {'active' if 0.55 <= proba <= 0.75 else ''}">Seuil 0.64</span>
                <span class="gauge-tick">0.75</span>
                <span class="gauge-tick">1.0</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Visualisation matplotlib ─────────────────
            st.markdown("""
            <div class="section-header" style="margin-top:1.75rem">
              <div class="section-dot"></div>
              <div class="section-title">Image redimensionnée (64×64)</div>
            </div>
            """, unsafe_allow_html=True)

            accent = "#34c759" if is_forest else "#ff9500"

            fig, ax = plt.subplots(figsize=(4, 4), facecolor="#040d08")
            ax.imshow(np.clip(image_test, 0, 1), interpolation='nearest')
            ax.set_facecolor("#040d08")

            for spine in ax.spines.values():
                spine.set_edgecolor(accent)
                spine.set_linewidth(1.5)

            ax.set_xticks([])
            ax.set_yticks([])

            patch = mpatches.Patch(
                color=accent,
                label=f"{class_txt}  ·  p = {proba:.4f}"
            )
            legend = ax.legend(
                handles=[patch],
                loc='lower right',
                fontsize=7,
                framealpha=0.7,
                facecolor="#040d08",
                edgecolor=accent,
                labelcolor=accent,
            )

            plt.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.error("Impossible de traiter l'image.")

    elif uploaded_file is not None and cnn_model is None:
        st.error("Le modèle CNN n'est pas disponible.")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="app-footer">
  <div class="footer-brand">Forest<span>Watch</span></div>
  <div class="footer-meta">CNN · Focal Loss · F1 · Seuil 0.64 · TensorFlow</div>
</div>
""", unsafe_allow_html=True)
