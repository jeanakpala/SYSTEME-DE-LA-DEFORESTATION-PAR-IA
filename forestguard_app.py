# ============================================================
# ForestGuard Togo — Application Streamlit
# Détection de déforestation par Deep Learning (CNN 7 canaux)
# Zones : Réserve d'Abdoulaye & Forêt de Fazao-Malfakassa
# Période : 2000-2024 | Résolution : 30 m (Sentinel-2/Landsat)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import io
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# --- Imports conditionnels ---
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
st.set_page_config(
    page_title="ForestGuard Togo",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_JSON      = Path(__file__).parent / "data" / "stats_forests.json"
MODEL_PATH     = Path(__file__).parent / "models" / "best_forest_cnn_colab.keras"
LIVRABLES_DIR  = Path(__file__).parent / "outputs" / "livrables"
SHAPEFILES_DIR = LIVRABLES_DIR / "shapefiles"
FIGURES_DIR    = Path(__file__).parent / "reports" / "figures"
TABLES_DIR     = Path(__file__).parent / "reports" / "tables"

# Couleurs thème forêt
VERT_FORET   = "#2d6a4f"
VERT_CLAIR   = "#52b788"
VERT_PALE    = "#b7e4c7"
ROUGE_ALERTE = "#d62728"
ORANGE_WARN  = "#ff7f0e"
BLANC        = "#ffffff"
GRIS_FOND    = "#f0f4f0"

# Pipeline preprocessing — identique à infer_local.py / app_local.py
CHANNEL_RANGES = {
    'red':  (0.00, 0.40), 'nir':  (0.00, 0.70), 'swir': (0.00, 0.55),
    'ndvi': (-0.20, 0.90), 'evi2': (-0.20, 0.80),
    'ndwi': (-0.60, 0.60), 'savi': (-0.20, 0.80),
}
LANDSAT_SCALE  = 0.0000275
LANDSAT_OFFSET = -0.2
EPS_SPEC       = 1e-10

CSS = f"""
<style>
  /* --- Fond général vert forêt --- */
  .stApp {{ background-color: #e8f5e9; }}

  /* --- Sidebar vert forêt foncé --- */
  section[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(170deg, {VERT_FORET} 0%, #1b4332 100%);
  }}
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label {{
    color: #d8f3dc !important;
  }}
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{
    color: {VERT_PALE} !important;
  }}
  section[data-testid="stSidebar"] .stDivider {{
    border-color: rgba(255,255,255,0.2) !important;
  }}

  /* --- Titres --- */
  h1, h2, h3 {{ color: #1b4332; }}

  /* --- Onglets --- */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: #c8e6c9;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: #1b4332;
    font-weight: 600;
    border-radius: 8px;
  }}
  .stTabs [aria-selected="true"] {{
    background-color: {VERT_FORET} !important;
    color: white !important;
  }}

  /* --- KPI cards --- */
  .kpi-card {{
      background: {BLANC}; border-left: 4px solid {VERT_FORET};
      border-radius: 8px; padding: 16px 20px; margin: 6px 0;
      box-shadow: 0 2px 8px rgba(27,67,50,.12);
  }}
  .kpi-val {{ font-size: 2rem; font-weight: 700; color: {VERT_FORET}; }}
  .kpi-lbl {{ font-size: 0.85rem; color: #555; }}

  /* --- Alert cards --- */
  .alert-card {{
      background: #fff3cd; border-left: 4px solid {ORANGE_WARN};
      border-radius: 8px; padding: 12px 16px; margin: 6px 0;
  }}
  .critical-card {{
      background: #fde8e8; border-left: 4px solid {ROUGE_ALERTE};
      border-radius: 8px; padding: 12px 16px; margin: 6px 0;
  }}

  /* --- Badge --- */
  .metric-badge {{
      display:inline-block; background:{VERT_FORET}; color:{BLANC};
      padding:4px 12px; border-radius:12px; font-size:0.8rem; margin:2px;
  }}

  /* --- Sidebar metric (fond semi-transparent sur vert foncé) --- */
  .sidebar-metric {{
      background: rgba(255,255,255,0.13);
      border-radius: 6px; padding: 8px 12px;
      margin: 4px 0; font-size: 0.82rem;
      border: 1px solid rgba(255,255,255,0.22);
      color: #d8f3dc !important;
  }}

  /* --- Boutons primaires --- */
  .stButton > button[kind="primary"],
  button[data-testid="stBaseButton-primary"] {{
    background-color: {VERT_FORET} !important;
    border-color: #1b4332 !important;
    color: white !important;
  }}
  .stButton > button[kind="primary"]:hover,
  button[data-testid="stBaseButton-primary"]:hover {{
    background-color: #1b4332 !important;
  }}

  /* --- Streamlit metric widget --- */
  [data-testid="metric-container"] {{
    background-color: rgba(255,255,255,0.75);
    border-radius: 8px;
    padding: 8px 12px;
  }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data
def load_data():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    if not TF_OK:
        return None, "TensorFlow non installé"
    if not MODEL_PATH.exists():
        return None, "Modèle absent (mode démo actif)"
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        return model, "Production"
    except Exception as e:
        return None, f"Erreur chargement: {e}"


DATA   = load_data()
MODEL, MODEL_STATUS = load_model()
MODE   = "Production" if MODEL is not None else "Démo"


# ============================================================
# UTILITAIRES — TRAITEMENT D'IMAGE
# ============================================================

def _normalize_channels(raw: np.ndarray) -> np.ndarray:
    """Normalise un stack (H, W, 7) selon CHANNEL_RANGES (identique à infer_local.py)."""
    keys = ['red', 'nir', 'swir', 'ndvi', 'evi2', 'ndwi', 'savi']
    out  = np.empty_like(raw, dtype=np.float32)
    for ci, ch in enumerate(keys):
        vmin, vmax = CHANNEL_RANGES[ch]
        out[:, :, ci] = np.clip(
            (raw[:, :, ci] - vmin) / max(vmax - vmin, EPS_SPEC), 0.0, 1.0
        )
    return out


def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """Construit un tenseur 256×256×7 depuis une image RGB (photo PNG/JPG).
    Mapping : R→Red, G→pseudo-NIR, B→pseudo-SWIR (identique à app_local.py).
    """
    from PIL import Image as PILImg
    pil = PILImg.fromarray(img_array.astype(np.uint8)).resize((256, 256), PILImg.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    red  = arr[:, :, 0]
    nir  = arr[:, :, 1]   # canal G → pseudo-NIR
    swir = arr[:, :, 2]   # canal B → pseudo-SWIR

    NDVI = (nir - red)  / (nir + red  + EPS_SPEC)
    EVI2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + EPS_SPEC)
    NDWI = (nir - swir) / (nir + swir + EPS_SPEC)
    SAVI = 1.5 * (nir - red) / (nir + red + 0.5 + EPS_SPEC)

    raw = np.stack([red, nir, swir, NDVI, EVI2, NDWI, SAVI], axis=-1)
    return np.expand_dims(_normalize_channels(raw), axis=0)   # (1, 256, 256, 7)


def compute_indices(img_array: np.ndarray) -> dict:
    """Calcule les 4 indices spectraux depuis une image RGB (photo)."""
    from PIL import Image as PILImg
    pil = PILImg.fromarray(img_array.astype(np.uint8)).resize((256, 256))
    arr = np.array(pil, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    red  = arr[:, :, 0]
    nir  = arr[:, :, 1]   # G → pseudo-NIR
    swir = arr[:, :, 2]   # B → pseudo-SWIR

    ndvi = float(np.mean((nir - red)  / (nir + red  + EPS_SPEC)))
    evi  = float(np.mean(2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + EPS_SPEC)))
    ndwi = float(np.mean((nir - swir) / (nir + swir + EPS_SPEC)))
    savi = float(np.mean(1.5 * (nir - red) / (nir + red + 0.5 + EPS_SPEC)))
    return {"NDVI": ndvi, "EVI2": evi, "NDWI": ndwi, "SAVI": savi}


def _process_geotiff(img_bytes: bytes) -> tuple:
    """Lit un GeoTIFF satellite (3 bandes: Red, NIR, SWIR) et retourne
    (img_display uint8, tensor (1,256,256,7), indices dict).
    Pipeline identique à infer_local.py / app_local.py.
    """
    from PIL import Image as PILImg
    with rasterio.open(io.BytesIO(img_bytes)) as src:
        sensor = 'landsat' if 'SR_B' in str(src.descriptions) else 'sentinel2'
        n = src.count
        b1 = src.read(1).astype(np.float32)
        b2 = src.read(2).astype(np.float32) if n >= 2 else b1.copy()
        b3 = src.read(3).astype(np.float32) if n >= 3 else b2.copy()

    # Conversion en réflectance de surface
    if sensor == 'landsat':
        b1 = b1 * LANDSAT_SCALE + LANDSAT_OFFSET
        b2 = b2 * LANDSAT_SCALE + LANDSAT_OFFSET
        b3 = b3 * LANDSAT_SCALE + LANDSAT_OFFSET
    else:
        b1 /= 10000.0; b2 /= 10000.0; b3 /= 10000.0

    red, nir, swir = b1, b2, b3
    valid = np.isfinite(red) & np.isfinite(nir) & np.isfinite(swir)

    def _s(a):
        return np.where(valid, a, np.nan).astype(np.float32)

    i_ndvi = _s((nir - red)  / (nir + red  + EPS_SPEC))
    i_evi2 = _s(2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + EPS_SPEC))
    i_ndwi = _s((nir - swir) / (nir + swir + EPS_SPEC))
    i_savi = _s(1.5 * (nir - red) / (nir + red + 0.5 + EPS_SPEC))

    raw = np.stack([red, nir, swir, i_ndvi, i_evi2, i_ndwi, i_savi], axis=-1).astype(np.float32)
    stack_norm = _normalize_channels(np.nan_to_num(raw, nan=0.0))

    # Resize chaque canal vers 256×256
    def _resize_ch(ch_2d):
        arr_u8 = (np.clip(ch_2d, 0, 1) * 255).astype(np.uint8)
        return np.array(
            PILImg.fromarray(arr_u8).resize((256, 256), PILImg.BILINEAR),
            dtype=np.float32
        ) / 255.0

    resized = np.stack([_resize_ch(stack_norm[:, :, c]) for c in range(7)], axis=-1)
    tensor  = np.expand_dims(resized, axis=0)  # (1, 256, 256, 7)

    # Image de visualisation : composite fausse couleur NIR-Red-SWIR
    r_ch = (np.clip(stack_norm[:, :, 1], 0, 1) * 255).astype(np.uint8)  # NIR → R
    g_ch = (np.clip(stack_norm[:, :, 0], 0, 1) * 255).astype(np.uint8)  # Red → G
    b_ch = (np.clip(stack_norm[:, :, 2], 0, 1) * 255).astype(np.uint8)  # SWIR → B
    img_display = np.stack([r_ch, g_ch, b_ch], axis=-1)

    indices = {
        "NDVI": float(np.nanmean(i_ndvi)),
        "EVI2": float(np.nanmean(i_evi2)),
        "NDWI": float(np.nanmean(i_ndwi)),
        "SAVI": float(np.nanmean(i_savi)),
    }
    return img_display, tensor, indices


def predict_image(tensor: np.ndarray, threshold: float) -> tuple[float, str]:
    """Retourne (probabilité, label). Mode démo si modèle absent."""
    if MODEL is not None:
        prob = float(MODEL.predict(tensor, verbose=0)[0][0])
    else:
        # Sigmoïde inversée sur le NDVI normalisé [0,1] (via CHANNEL_RANGES) :
        #   norm_NDVI ≈ 0.8 (forêt dense)  → ~8%  proba déforestation
        #   norm_NDVI ≈ 0.5 (mixte)        → ~50%
        #   norm_NDVI ≈ 0.2 (sol nu)       → ~90%
        mean_ndvi = float(np.mean(tensor[0, :, :, 3]))
        base = 1.0 / (1.0 + np.exp(7.5 * (mean_ndvi - 0.50)))
        noise = float(np.random.normal(0, 0.04))
        prob = float(np.clip(base + noise, 0.03, 0.97))
    label = "Déforestation" if prob >= threshold else "Forêt"
    return prob, label


def build_heatmap(tensor: np.ndarray) -> np.ndarray:
    """Génère une heatmap de suspicion 256×256 depuis le canal NDVI."""
    ndvi_map = tensor[0, :, :, 3]   # canal 3 = NDVI
    heat = np.clip(0.5 - ndvi_map, 0, 1)
    return heat


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;font-size:2.5rem;'>🇹🇬</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h2 style='text-align:center;color:{VERT_FORET};margin:0;'>ForestGuard Togo</h2>",
            unsafe_allow_html=True
        )
        st.caption("Détection de déforestation par Deep Learning")
        st.divider()

        # Paramètres globaux
        st.subheader("⚙️ Paramètres")
        zone = st.selectbox(
            "Zone d'étude",
            ["Toutes les zones", "Abdoulaye", "Fazao"],
            help="Filtre les analyses par site"
        )
        threshold = st.slider(
            "Seuil de classification",
            0.0, 1.0, 0.10, 0.05,
            help="Seuil optimal OS3 : 0.10 (best_epoch=6, 18 epochs)"
        )

        st.divider()

        # Métriques réelles du modèle (OS3)
        st.subheader("🎯 Métriques CNN (OS3)")
        m = DATA["modele"]["metriques"]
        e = DATA["modele"]["entrainement"]
        st.markdown(
            f"""
            <div class="sidebar-metric">Accuracy  : <b>{m['accuracy']*100:.2f}%</b></div>
            <div class="sidebar-metric">Precision : <b>{m['precision']*100:.1f}%</b></div>
            <div class="sidebar-metric">Recall    : <b>{m['recall']*100:.1f}%</b></div>
            <div class="sidebar-metric">F1-Score  : <b>{m['f1_score']*100:.2f}%</b></div>
            <div class="sidebar-metric">IoU       : <b>{m['iou_jaccard']*100:.1f}%</b></div>
            <div class="sidebar-metric">AUC-ROC   : <b>{m['auc_roc']:.4f}</b></div>
            <div class="sidebar-metric">Seuil opt.: <b>{DATA['modele']['seuil_optimal']}</b></div>
            <div class="sidebar-metric">Epochs    : <b>{e['epochs_effectifs']}/{e['epochs_max']} (best={e['meilleure_epoch']})</b></div>
            <div class="sidebar-metric">Canaux    : <b>{DATA['modele']['canaux']} (7-ch)</b></div>
            <div class="sidebar-metric">Patches   : <b>{m['nb_patches_total']:,}</b></div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # Statut mode
        mode_color = VERT_FORET if MODE == "Production" else ORANGE_WARN
        st.markdown(
            f"<div style='background:{mode_color};color:#fff;border-radius:6px;"
            f"padding:8px;text-align:center;font-weight:600;'>"
            f"Mode : {MODE}</div>",
            unsafe_allow_html=True
        )
        st.caption(MODEL_STATUS)

        st.divider()
        st.caption(
            "Mémoire de Master 2026\n\n"
            "Dép. Informatique — Université de l'Afrique de l'Ouest (UCAO-UUT)\n\n"
            f"Données : Sentinel-2 / Landsat | {DATA['metadata']['periode']}"
        )

    return zone, threshold


# ============================================================
# ONGLET 1 — ANALYSE D'IMAGE
# ============================================================
def tab_analyse(threshold: float, zone: str):
    st.header("🔍 Analyse d'Image Satellite")
    st.markdown(
        "Chargez une image satellite (RGB ou GeoTIFF multi-bandes) pour obtenir "
        "une prédiction forêt/déforestation, les indices spectraux et une heatmap de suspicion."
    )

    uploaded_files = st.file_uploader(
        "Sélectionner une ou plusieurs images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        help="Formats acceptés : PNG, JPG, JPEG, TIF, TIFF, GeoTIFF"
    )

    if not uploaded_files:
        st.info(
            "Aucune image chargée. "
            "En mode **Démo**, les prédictions sont simulées à partir des statistiques du mémoire."
        )
        _show_demo_analysis(threshold)
        return

    for uploaded in uploaded_files:
        st.divider()
        st.subheader(f"📁 {uploaded.name}")
        _analyse_single_file(uploaded, threshold)


def _show_demo_analysis(threshold: float):
    """Affiche un exemple de résultat simulé."""
    np.random.seed(42)
    # Génération d'une image démo verte (forêt)
    demo_img = np.random.randint(30, 120, (256, 256, 3), dtype=np.uint8)
    demo_img[:, :, 1] = np.clip(demo_img[:, :, 1] + 80, 0, 255)  # canal vert renforcé

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Image démo (synthétique)**")
        from PIL import Image as PILImg
        st.image(PILImg.fromarray(demo_img), caption="Image de démonstration", use_container_width=True)

    tensor  = preprocess_image(demo_img)
    prob, label = predict_image(tensor, threshold)
    indices = compute_indices(demo_img)

    with col2:
        _display_prediction(prob, label, threshold)
        st.caption("*(Prédiction simulée — mode démo)*")

    _display_indices(indices)


def _analyse_single_file(uploaded, threshold: float):
    """Analyse complète d'un fichier uploadé."""
    img_bytes = uploaded.read()
    img_array = None
    tensor    = None
    indices   = None

    # ── GeoTIFF : pipeline satellite complet (Red/NIR/SWIR → 7 canaux) ──────
    if uploaded.name.lower().endswith((".tif", ".tiff")) and RASTERIO_OK:
        try:
            img_array, tensor, indices = _process_geotiff(img_bytes)
        except Exception as e:
            st.warning(f"Pipeline satellite échoué ({e}). Bascule en mode RGB…")
            img_array = None

    # ── Fallback PIL (PNG/JPG ou GeoTIFF illisible) ───────────────────────────
    if tensor is None and PIL_OK:
        try:
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(pil_img)
        except Exception as e:
            st.error(f"Impossible de lire l'image : {e}")
            return
        if img_array is None:
            st.error("Pillow et rasterio non disponibles — impossible de lire l'image.")
            return
        tensor  = preprocess_image(img_array)
        indices = compute_indices(img_array)

    if tensor is None:
        st.error("Impossible de construire le tenseur d'analyse.")
        return

    prob, label = predict_image(tensor, threshold)
    heat        = build_heatmap(tensor)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("**Image originale**")
        from PIL import Image as PILImg
        pil_disp = PILImg.fromarray(img_array).resize((256, 256))
        st.image(pil_disp, use_container_width=True)

    with col2:
        st.markdown("**Résultat de classification**")
        _display_prediction(prob, label, threshold)

    with col3:
        st.markdown("**Heatmap de suspicion**")
        _display_heatmap(heat)

    # Indices spectraux
    st.markdown("---")
    _display_indices(indices)

    # Détails techniques
    with st.expander("🔧 Détails techniques — Pipeline de traitement", expanded=False):
        st.markdown(
            f"""
**Pipeline 5 phases (Mémoire Master 2026)**

| Phase | Description | Statut |
|-------|-------------|--------|
| OS1 | Collecte & Préparation (Sentinel-2/Landsat) | ✅ Complété |
| OS2 | Extraction features spectrales (7 canaux) | ✅ Complété |
| OS3 | Entraînement CNN (256×256×7, 2 000 patches) | ✅ Accuracy=99.75%, AUC=1.00 |
| OS4 | Inférence sliding window (STRIDE=128) | ✅ Exécuté (anomalie capteur détectée) |
| OS5 | Livrables opérationnels (SHP/GeoTIFF/PDF/HTML) | 🔄 En cours |

**Architecture CNN :** {DATA['modele']['architecture']}

**Canaux d'entrée :** {', '.join(DATA['modele']['noms_canaux'])}

**Prétraitement appliqué :**
- Redimensionnement → 256×256 px
- Simulation NIR/SWIR à partir du RGB
- Calcul NDVI, EVI2, NDWI, SAVI
- Stack 7 canaux float32, normalisé [0, 1]
- Seuil de classification : **{threshold:.2f}** (optimal OS3 : 0.10)
            """
        )


def _display_prediction(prob: float, label: str, threshold: float):
    """Affiche jauge de confiance et label."""
    is_deforest = (label == "Déforestation")
    color = ROUGE_ALERTE if is_deforest else VERT_FORET
    icon  = "🌲" if not is_deforest else "⚠️"

    st.markdown(
        f"""
        <div style="background:{color};color:#fff;border-radius:10px;
        padding:20px;text-align:center;margin-bottom:12px;">
            <div style="font-size:2.5rem;">{icon}</div>
            <div style="font-size:1.4rem;font-weight:700;">{label}</div>
            <div style="font-size:1.1rem;">Probabilité : {prob:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if PLOTLY_OK:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": VERT_PALE},
                    {"range": [threshold * 100, 100], "color": "#fde8e8"},
                ],
                "threshold": {
                    "line": {"color": ORANGE_WARN, "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
            title={"text": "Indice de déforestation"},
        ))
        fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


def _display_heatmap(heat: np.ndarray):
    """Affiche la heatmap de suspicion."""
    if PLOTLY_OK:
        fig = px.imshow(
            heat, color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=1,
            labels={"color": "Suspicion"},
            title="Zones à risque (rouge = déforestation probable)"
        )
        fig.update_layout(
            height=280, margin=dict(t=40, b=0, l=0, r=0),
            coloraxis_showscale=True
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.image(heat, caption="Heatmap", use_container_width=True)


def _display_indices(indices: dict):
    """Affiche les 4 indices spectraux avec interprétation."""
    st.markdown("**Indices spectraux calculés**")
    cols = st.columns(4)
    interpretations = {
        "NDVI": {
            "seuils": [(-1, 0.1, "Sol nu / eau", ROUGE_ALERTE),
                       (0.1, 0.4, "Végétation clairsemée", ORANGE_WARN),
                       (0.4, 1.1, "Végétation dense (forêt)", VERT_FORET)],
        },
        "EVI2": {
            "seuils": [(-1, 0.1, "Zone dégradée", ROUGE_ALERTE),
                       (0.1, 0.35, "Végétation modérée", ORANGE_WARN),
                       (0.35, 1.1, "Forêt dense", VERT_FORET)],
        },
        "NDWI": {
            "seuils": [(-1, -0.1, "Végétation sèche", ORANGE_WARN),
                       (-0.1, 0.2, "Humidité modérée", VERT_CLAIR),
                       (0.2, 1.1, "Surface en eau / humide", "#1f77b4")],
        },
        "SAVI": {
            "seuils": [(-1, 0.1, "Sol nu", ROUGE_ALERTE),
                       (0.1, 0.35, "Végétation éparse", ORANGE_WARN),
                       (0.35, 1.1, "Végétation dense", VERT_FORET)],
        },
    }
    for col, (name, val) in zip(cols, indices.items()):
        interp_txt, interp_color = "—", GRIS_FOND
        for lo, hi, txt, c in interpretations[name]["seuils"]:
            if lo <= val < hi:
                interp_txt, interp_color = txt, c
                break
        with col:
            st.markdown(
                f"""
                <div style="background:#fff;border-radius:8px;padding:14px;
                border-left:4px solid {interp_color};text-align:center;
                box-shadow:0 1px 4px rgba(0,0,0,.08);">
                    <div style="font-size:1.1rem;font-weight:700;color:{VERT_FORET};">{name}</div>
                    <div style="font-size:1.6rem;font-weight:700;">{val:.3f}</div>
                    <div style="font-size:0.78rem;color:{interp_color};font-weight:600;">{interp_txt}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ONGLET 2 — TABLEAU DE BORD
# ============================================================
def tab_dashboard(zone: str):
    st.header("📊 Tableau de Bord — Résultats OS4 (2000–2024)")

    kpis  = DATA["kpis"]
    sites = DATA["sites"]

    # --- Bannière anomalie capteur ---
    st.markdown(
        f"""
        <div style="background:#fff3cd;border-left:5px solid {ORANGE_WARN};
        border-radius:8px;padding:14px 18px;margin-bottom:16px;">
        <b>⚠️ Anomalie capteur détectée (OS4)</b> — Les années <b>2015 et 2024</b>
        (Sentinel-2) présentent une classification dégénérée : le modèle CNN entraîné
        sur Landsat classe tous les pixels comme non-forêt sur les images Sentinel-2
        en raison d'un décalage radiométrique inter-capteur non corrigé.
        Les résultats fiables sont : <b>Abdoulaye 2000/2005/2013/2020</b> et
        <b>Fazao 2013/2020</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- KPIs (données valides uniquement) ---
    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, "Superficie initiale", f"{kpis['superficie_initiale_ha']:,} ha",
         "🌿", "Abdoulaye 2000 + Fazao 2013")
    _kpi(c2, "Perte Fazao 2013→2020", f"{kpis['perte_totale_ha']:,} ha",
         "📉", "Seule période fiable homogène", color=ORANGE_WARN)
    _kpi(c3, "Superficie valide 2020", f"{kpis['superficie_valide_2020_ha']:,} ha",
         "🗺️", "Abdoulaye + Fazao (dernières années valides)")
    _kpi(c4, "Zone la plus menacée", kpis["foret_plus_menacee"],
         "⚠️", f"−42 648 ha (2013→2020, {kpis['taux_deforestation_moyen_pct']}%/an)",
         color=ROUGE_ALERTE)

    st.divider()

    # --- Graphique évolution (années valides uniquement) ---
    st.subheader("Évolution de la couverture forestière — années valides (Landsat)")

    rows_evol = []
    for site_name, site_data in sites.items():
        if zone not in ("Toutes les zones", site_name):
            continue
        for yr_str, val in site_data["superficie_foret_ha"].items():
            yr = int(yr_str)
            if val is not None:   # ignorer les anomalies (null)
                capteur = site_data["capteur_par_annee"].get(yr_str, "?")
                rows_evol.append({
                    "Année": yr, "Superficie (ha)": val,
                    "Site": site_name, "Capteur": capteur
                })

    df_evol = pd.DataFrame(rows_evol)

    if PLOTLY_OK and not df_evol.empty:
        fig = go.Figure()
        for site_name, grp in df_evol.groupby("Site"):
            color = ROUGE_ALERTE if site_name == "Abdoulaye" else VERT_FORET
            fig.add_trace(go.Scatter(
                x=grp["Année"], y=grp["Superficie (ha)"],
                mode="lines+markers+text",
                name=site_name,
                line=dict(color=color, width=2.5),
                marker=dict(size=10),
                text=[f"{v/1000:.0f}k" for v in grp["Superficie (ha)"]],
                textposition="top center",
                hovertemplate="%{x} : %{y:,.0f} ha<extra>" + site_name + "</extra>"
            ))
        fig.update_layout(
            plot_bgcolor=BLANC, paper_bgcolor=BLANC,
            height=380, legend_title_text="Site",
            yaxis_title="Superficie forestière (ha)",
            xaxis_title="Année",
            title="Superficie forestière par site — années valides (Landsat uniquement)",
            annotations=[dict(
                x=0.5, y=-0.18, xref="paper", yref="paper",
                text="⚠️ Années 2015 et 2024 (Sentinel-2) exclues — anomalie radiométrique inter-capteur",
                showarrow=False, font=dict(size=10, color="#888"), align="center"
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- NDVI temporel (toutes années) ---
    st.subheader("Évolution NDVI temporel — tous capteurs")
    rows_ndvi = []
    ndvi_data = DATA.get("indices_spectraux_temporels", {})
    for site_name, annees in ndvi_data.items():
        if zone not in ("Toutes les zones", site_name):
            continue
        for yr_str, vals in annees.items():
            rows_ndvi.append({
                "Année": int(yr_str),
                "NDVI moyen": vals["NDVI"],
                "Site": site_name,
                "Capteur": vals["capteur"],
                "Anomalie": yr_str in [str(a) for a in sites[site_name].get("annees_anomalie", [])]
            })
    df_ndvi = pd.DataFrame(rows_ndvi)

    if PLOTLY_OK and not df_ndvi.empty:
        fig_ndvi = go.Figure()
        for site_name, grp in df_ndvi.groupby("Site"):
            color = ROUGE_ALERTE if site_name == "Abdoulaye" else VERT_FORET
            grp_val  = grp[~grp["Anomalie"]]
            grp_anom = grp[grp["Anomalie"]]
            fig_ndvi.add_trace(go.Scatter(
                x=grp_val["Année"], y=grp_val["NDVI moyen"],
                mode="lines+markers", name=site_name,
                line=dict(color=color, width=2),
                marker=dict(size=9),
                hovertemplate="%{x} : NDVI=%{y:.3f} (Landsat)<extra>" + site_name + "</extra>"
            ))
            if not grp_anom.empty:
                fig_ndvi.add_trace(go.Scatter(
                    x=grp_anom["Année"], y=grp_anom["NDVI moyen"],
                    mode="markers", name=f"{site_name} ⚠️ Sentinel-2",
                    marker=dict(size=11, symbol="x", color=ORANGE_WARN),
                    hovertemplate="%{x} : NDVI=%{y:.3f} (Sentinel-2 — à calibrer)<extra></extra>"
                ))
        fig_ndvi.update_layout(
            plot_bgcolor=BLANC, paper_bgcolor=BLANC, height=300,
            yaxis_title="NDVI moyen", xaxis_title="Année",
            title="NDVI moyen par site (× = années Sentinel-2 non calibrées)"
        )
        st.plotly_chart(fig_ndvi, use_container_width=True)

    # --- Carte + Taux ---
    col_bar, col_map = st.columns([1, 1])

    with col_bar:
        st.subheader("Taux de déforestation (périodes fiables)")
        rows_taux = []
        for site_name, site_data in sites.items():
            if zone not in ("Toutes les zones", site_name):
                continue
            for periode, taux in site_data["taux_annuel_pct"].items():
                rows_taux.append({"Période": periode, "Taux (%/an)": taux, "Site": site_name})
        df_taux = pd.DataFrame(rows_taux)

        if PLOTLY_OK and not df_taux.empty:
            fig2 = px.bar(
                df_taux, x="Période", y="Taux (%/an)", color="Site",
                barmode="group",
                color_discrete_map={"Abdoulaye": ROUGE_ALERTE, "Fazao": VERT_FORET},
                title="Taux annuel de déforestation (périodes valides)",
                text_auto=".2f"
            )
            fig2.update_layout(
                plot_bgcolor=BLANC, paper_bgcolor=BLANC,
                height=300, xaxis_tickangle=-20
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col_map:
        st.subheader("Carte interactive — Togo")
        if FOLIUM_OK:
            m = folium.Map(location=[8.70, 1.07], zoom_start=8,
                           tiles="CartoDB positron")
            for site_name, site_data in sites.items():
                if zone not in ("Toutes les zones", site_name):
                    continue
                lat = site_data["coordonnees"]["lat"]
                lon = site_data["coordonnees"]["lon"]
                couleur = "#d62728" if site_name == "Abdoulaye" else "#ff7f0e"
                # Dernière superficie valide
                valides = {k: v for k, v in site_data["superficie_foret_ha"].items() if v is not None}
                sup_valide = list(valides.values())[-1] if valides else 0
                annee_valide = list(valides.keys())[-1] if valides else "N/A"
                perte = sum(site_data["deforesstation_ha"].values())
                popup_html = (
                    f"<b>{site_data['nom_complet']}</b><br>"
                    f"Superficie {annee_valide} : {sup_valide:,.0f} ha<br>"
                    f"Perte cumulée : {perte:,.0f} ha<br>"
                    f"Menace : <b>{site_data['menace']}</b><br>"
                    f"<i>⚠️ 2015/2024 : anomalie Sentinel-2</i>"
                )
                rayon = max(8, int(sup_valide / 30000))
                folium.CircleMarker(
                    location=[lat, lon], radius=rayon,
                    color=couleur, fill=True, fill_color=couleur,
                    fill_opacity=0.35, weight=2,
                    popup=folium.Popup(popup_html, max_width=270),
                    tooltip=f"{site_name} — {site_data['menace']}"
                ).add_to(m)
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(
                        color="red" if site_name == "Abdoulaye" else "orange",
                        icon="leaf", prefix="glyphicon"
                    ),
                    popup=folium.Popup(popup_html, max_width=270),
                    tooltip=site_name
                ).add_to(m)
            st_folium(m, height=300, use_container_width=True)
        else:
            st.info("streamlit-folium non installé.")

    st.divider()

    # --- Tableau OS4 complet (avec annotations anomalie) ---
    st.subheader("Résultats OS4 — Inférence complète (toutes images)")
    rows_os4 = []
    df_os4_src = pd.read_csv(
        Path(__file__).parent / "reports" / "tables" / "deforestation_stats.csv"
    ) if (Path(__file__).parent / "reports" / "tables" / "deforestation_stats.csv").exists() else None

    if df_os4_src is not None:
        for _, row in df_os4_src.iterrows():
            site = row["site"].capitalize()
            yr   = int(row["year"])
            anomalie = yr in sites.get(site, {}).get("annees_anomalie", [])
            rows_os4.append({
                "Site": site,
                "Année": yr,
                "Capteur": DATA["sites"].get(site, {}).get("capteur_par_annee", {}).get(str(yr), "?"),
                "Forêt (ha)": f"{row['forest_ha']:,.1f}" if pd.notna(row["forest_ha"]) else "—",
                "Déforesté (ha)": f"{row['deforest_ha']:,.1f}" if pd.notna(row.get("deforest_ha")) else "—",
                "Reboisé (ha)": f"{row['reforest_ha']:,.1f}" if pd.notna(row.get("reforest_ha")) else "—",
                "% Déforesté": f"{row['deforest_pct']:.1f}%" if pd.notna(row.get("deforest_pct")) else "—",
                "Statut": "⚠️ Anomalie capteur" if anomalie else "✅ Valide",
            })
        df_os4_tab = pd.DataFrame(rows_os4)
        st.dataframe(df_os4_tab, hide_index=True, use_container_width=True)
    else:
        # Fallback tableau manuel
        rows_tab = []
        for site_name, site_data in sites.items():
            if zone not in ("Toutes les zones", site_name):
                continue
            for yr_str, val in site_data["superficie_foret_ha"].items():
                yr = int(yr_str)
                anomalie = yr in site_data.get("annees_anomalie", [])
                rows_tab.append({
                    "Site": site_name,
                    "Année": yr,
                    "Capteur": site_data["capteur_par_annee"].get(yr_str, "?"),
                    "Forêt (ha)": f"{val:,.1f}" if val else "⚠️ 0 (anomalie)",
                    "Statut": "⚠️ Anomalie capteur" if anomalie else "✅ Valide",
                })
        st.dataframe(pd.DataFrame(rows_tab), hide_index=True, use_container_width=True)

    # --- Livrables OS4 générés ---
    st.divider()
    st.subheader("📦 Livrables OS4 générés")
    liv_path = Path(__file__).parent / "outputs" / "livrables"
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown("**Shapefiles / GeoJSON**")
        for shp in DATA.get("livrables_generes", {}).get("shapefiles", []):
            st.markdown(f"- `{shp}`")
    with col_l2:
        st.markdown("**Rapports**")
        pdf_path = liv_path / "rapport_deforestation.pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Rapport PDF (OS4)",
                    data=f.read(), file_name="rapport_deforestation.pdf",
                    mime="application/pdf", use_container_width=True
                )
        xlsx_path = liv_path / "rapport_deforestation.xlsx"
        if xlsx_path.exists():
            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "⬇️ Rapport Excel (OS4)",
                    data=f.read(), file_name="rapport_deforestation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    with col_l3:
        st.markdown("**Carte interactive OS4**")
        html_path = liv_path / "carte_deforestation_interactive.html"
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "⬇️ Carte HTML (Folium OS4)",
                    data=f.read().encode("utf-8"),
                    file_name="carte_deforestation_interactive.html",
                    mime="text/html", use_container_width=True
                )


def _kpi(col, label, value, icon, note="", color=VERT_FORET):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color:{color}">
                <div style="font-size:1.6rem;">{icon}</div>
                <div class="kpi-val" style="color:{color};">{value}</div>
                <div class="kpi-lbl">{label}</div>
                <div style="font-size:0.75rem;color:#888;margin-top:4px;">{note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ONGLET 3 — COMPARAISON MULTI-IMAGES
# ============================================================
def tab_comparaison(threshold: float):
    st.header("🔬 Comparaison Multi-Images")
    st.markdown("Chargez 2 à 4 images pour comparer les prédictions et les indices spectraux.")

    files = st.file_uploader(
        "Sélectionner 2 à 4 images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="multi_upload"
    )

    if not files:
        st.info("Aucune image chargée. Chargez entre 2 et 4 images.")
        return

    if len(files) > 4:
        st.warning("Maximum 4 images. Seules les 4 premières seront analysées.")
        files = files[:4]

    results = []
    cols = st.columns(min(len(files), 2))

    for i, f in enumerate(files):
        img_bytes = f.read()
        img_array = None

        img_array = None
        tensor    = None
        indices   = None

        if f.name.lower().endswith((".tif", ".tiff")) and RASTERIO_OK:
            try:
                img_array, tensor, indices = _process_geotiff(img_bytes)
            except Exception:
                img_array = None

        if tensor is None and PIL_OK:
            try:
                img_array = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            except Exception:
                st.error(f"Erreur lecture : {f.name}")
                continue
            if img_array is None:
                continue
            tensor  = preprocess_image(img_array)
            indices = compute_indices(img_array)

        if tensor is None:
            continue

        prob, label = predict_image(tensor, threshold)

        col_idx = i % 2
        with cols[col_idx]:
            from PIL import Image as PILImg
            pil_disp = PILImg.fromarray(img_array).resize((240, 240))
            badge_color = ROUGE_ALERTE if label == "Déforestation" else VERT_FORET
            st.image(pil_disp, caption=f.name, use_container_width=True)
            st.markdown(
                f"<div style='background:{badge_color};color:#fff;border-radius:6px;"
                f"padding:8px;text-align:center;'>"
                f"<b>{label}</b> — {prob:.1%}</div>",
                unsafe_allow_html=True
            )

        results.append({
            "Image": f.name,
            "Prédiction": label,
            "Probabilité": f"{prob:.1%}",
            "NDVI": f"{indices['NDVI']:.3f}",
            "EVI2": f"{indices['EVI2']:.3f}",
            "NDWI": f"{indices['NDWI']:.3f}",
            "SAVI": f"{indices['SAVI']:.3f}",
        })

    if results:
        st.divider()
        st.subheader("Tableau comparatif")
        df_comp = pd.DataFrame(results)
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

        # Analyse différentielle NDVI si 2 images
        if len(results) == 2:
            st.divider()
            st.subheader("Analyse différentielle NDVI")
            ndvi1 = float(results[0]["NDVI"])
            ndvi2 = float(results[1]["NDVI"])
            delta = ndvi2 - ndvi1
            if abs(delta) < 0.05:
                statut, color = "Stable", VERT_FORET
            elif delta < 0:
                statut, color = "Dégradation", ROUGE_ALERTE
            else:
                statut, color = "Régénération", "#1f77b4"
            st.markdown(
                f"""
                <div style="background:{color};color:#fff;border-radius:8px;
                padding:16px;text-align:center;">
                    <b>Changement ΔNDVI = {delta:+.3f}</b> → <b>{statut}</b>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ONGLET 3b — CARTOGRAPHIE OS5
# ============================================================
def tab_cartographie(zone: str):
    st.header("🗺️ Cartographie OS5 — Zones Forestières & Changements")

    # --- Cartes de changement (PNG OS4) ---
    st.subheader("Cartes de changement forestier (OS4 — années valides Landsat)")
    col_a, col_f = st.columns(2)
    with col_a:
        p = FIGURES_DIR / "change_maps_abdoulaye.png"
        if p.exists():
            st.image(str(p), caption="Réserve d'Abdoulaye — changements 2000/2005/2013/2020",
                     use_container_width=True)
        else:
            st.info("change_maps_abdoulaye.png introuvable")
    with col_f:
        p = FIGURES_DIR / "change_maps_fazao.png"
        if p.exists():
            st.image(str(p), caption="Fazao-Malfakassa — changements 2013/2020",
                     use_container_width=True)
        else:
            st.info("change_maps_fazao.png introuvable")

    p_tl = FIGURES_DIR / "forest_area_timeline.png"
    if p_tl.exists():
        st.image(str(p_tl), caption="Évolution temporelle — superficie forestière (ha)",
                 use_container_width=True)

    st.divider()

    # --- Carte Folium interactive avec GeoJSON ---
    st.subheader("Carte interactive — Zones forestières par année (GeoJSON OS5)")

    geojson_all = SHAPEFILES_DIR / "forets_tous_sites_toutes_annees.geojson"

    if FOLIUM_OK and geojson_all.exists():
        import json as _json
        with open(geojson_all, encoding="utf-8") as f:
            gj = _json.load(f)

        # Filtrer par zone si besoin
        if zone != "Toutes les zones":
            site_key = zone.lower()
            gj["features"] = [ft for ft in gj["features"]
                               if ft["properties"].get("site", "") == site_key]

        year_colors = {
            2000: "#1a9850", 2005: "#52b788", 2013: "#b7e4c7",
            2020: "#ffd166", 2015: "#f4a261", 2024: "#e63946"
        }

        m = folium.Map(location=[8.85, 1.10], zoom_start=8, tiles="CartoDB positron")

        folium.GeoJson(
            gj,
            name="Zones forestières OS5",
            style_function=lambda feat: {
                "fillColor": year_colors.get(feat["properties"].get("year", 0), VERT_FORET),
                "color": "#1b4332",
                "weight": 1.5,
                "fillOpacity": 0.55,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["site", "year", "area_ha"],
                aliases=["Site", "Année", "Superficie (ha)"],
                localize=True
            ),
        ).add_to(m)

        # Légende années
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
             background:white;padding:10px;border-radius:8px;font-size:12px;
             border:1px solid #ccc;box-shadow:2px 2px 6px rgba(0,0,0,.2);">
          <b>Année (zones valides)</b><br>
          <span style="color:#1a9850">&#9632;</span> 2000 &nbsp;
          <span style="color:#52b788">&#9632;</span> 2005 &nbsp;
          <span style="color:#b7e4c7">&#9632;</span> 2013<br>
          <span style="color:#ffd166">&#9632;</span> 2020 &nbsp;
          <span style="color:#f4a261">&#9632;</span> 2015* &nbsp;
          <span style="color:#e63946">&#9632;</span> 2024*<br>
          <small>* anomalie Sentinel-2</small>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=520)
    elif not FOLIUM_OK:
        st.warning("Folium non installé : `pip install folium streamlit-folium`")
    else:
        st.warning(f"GeoJSON introuvable : {geojson_all}")

    st.divider()

    # --- Carte HTML pré-générée OS5 ---
    carte_html = LIVRABLES_DIR / "carte_deforestation_interactive.html"
    if carte_html.exists():
        st.subheader("Carte déforestation interactive complète (OS5 — HTML généré)")
        with open(carte_html, encoding="utf-8") as f:
            html_content = f.read()
        import streamlit.components.v1 as components
        components.html(html_content, height=600, scrolling=True)
        st.divider()

    # --- Téléchargement des GeoJSON/SHP ---
    st.subheader("Télécharger les livrables cartographiques")
    st.markdown("**GeoJSON par site et par année (années valides Landsat)**")

    dl_files = {
        "Tous sites / toutes années": SHAPEFILES_DIR / "forets_tous_sites_toutes_annees.geojson",
        "Abdoulaye 2000": SHAPEFILES_DIR / "foret_abdoulaye_2000.geojson",
        "Abdoulaye 2005": SHAPEFILES_DIR / "foret_abdoulaye_2005.geojson",
        "Abdoulaye 2013": SHAPEFILES_DIR / "foret_abdoulaye_2013.geojson",
        "Abdoulaye 2020": SHAPEFILES_DIR / "foret_abdoulaye_2020.geojson",
        "Fazao 2013":     SHAPEFILES_DIR / "foret_fazao_2013.geojson",
        "Fazao 2020":     SHAPEFILES_DIR / "foret_fazao_2020.geojson",
    }

    cols = st.columns(4)
    for i, (label, path) in enumerate(dl_files.items()):
        with cols[i % 4]:
            if path.exists():
                with open(path, "rb") as f:
                    st.download_button(
                        f"⬇️ {label}",
                        data=f.read(),
                        file_name=path.name,
                        mime="application/json",
                        use_container_width=True,
                        key=f"dl_geojson_{i}"
                    )
            else:
                st.caption(f"❌ {label}")


# ============================================================
# ONGLET 4 — RAPPORTS PDF
# ============================================================
def tab_rapports():
    st.header("📄 Génération de Rapports PDF")

    if not REPORTLAB_OK:
        st.error(
            "ReportLab non installé. Exécuter : `pip install reportlab`\n\n"
            "Une fois installé, rechargez la page."
        )
        return

    col_opt, col_prev = st.columns([1, 1])

    with col_opt:
        type_rapport = st.selectbox(
            "Type de rapport",
            ["Rapport technique complet", "Rapport par zone",
             "Note de synthèse (non-technique)", "Fiche prédiction image"]
        )
        periode = st.selectbox(
            "Période temporelle",
            ["2000-2024 (complète)", "2013-2024", "2020-2024"]
        )
        zone_pdf = st.selectbox(
            "Zone", ["Toutes les zones", "Abdoulaye", "Fazao"], key="zone_pdf"
        )
        st.markdown("**Contenu à inclure**")
        inc_graphiques  = st.checkbox("Graphiques statistiques", value=True)
        inc_tableaux    = st.checkbox("Tableaux de données", value=True)
        inc_methodo     = st.checkbox("Méthodologie", value=True)
        inc_recommandations = st.checkbox("Recommandations", value=True)

    with col_prev:
        st.markdown("**Aperçu du contenu**")
        st.markdown(
            f"""
            - **Type :** {type_rapport}
            - **Période :** {periode}
            - **Zone :** {zone_pdf}
            - **Graphiques :** {'✅' if inc_graphiques else '❌'}
            - **Tableaux :** {'✅' if inc_tableaux else '❌'}
            - **Méthodologie :** {'✅' if inc_methodo else '❌'}
            - **Recommandations :** {'✅' if inc_recommandations else '❌'}
            """
        )

        date_str = datetime.now().strftime("%Y%m%d")
        type_slug = type_rapport.split()[0].lower()
        pdf_name  = f"ForestGuard_Togo_Rapport_{date_str}_{type_slug}.pdf"
        st.info(f"Nom du fichier : **{pdf_name}**")

        if st.button("🖨️ Générer le rapport PDF", type="primary", use_container_width=True):
            with st.spinner("Génération du PDF en cours…"):
                pdf_bytes = _generate_pdf(
                    type_rapport, periode, zone_pdf,
                    inc_graphiques, inc_tableaux, inc_methodo, inc_recommandations
                )
            if pdf_bytes:
                st.success("Rapport généré avec succès !")
                st.download_button(
                    label="⬇️ Télécharger le PDF",
                    data=pdf_bytes,
                    file_name=pdf_name,
                    mime="application/pdf",
                    use_container_width=True
                )

    st.divider()

    # --- Livrables OS5 pré-générés ---
    st.subheader("Livrables OS5 pré-générés")
    st.markdown("Fichiers produits par les notebooks OS4/OS5 (disponibles immédiatement).")

    col1, col2, col3 = st.columns(3)
    with col1:
        p_xlsx = LIVRABLES_DIR / "rapport_deforestation.xlsx"
        if p_xlsx.exists():
            with open(p_xlsx, "rb") as f:
                st.download_button("⬇️ Rapport XLSX (OS5)", f.read(),
                                   "rapport_deforestation.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
        else:
            st.caption("❌ rapport_deforestation.xlsx absent")
    with col2:
        p_pdf = LIVRABLES_DIR / "rapport_deforestation.pdf"
        if p_pdf.exists():
            with open(p_pdf, "rb") as f:
                st.download_button("⬇️ Rapport PDF (OS5)", f.read(),
                                   "rapport_deforestation.pdf", "application/pdf",
                                   use_container_width=True)
        else:
            st.caption("❌ rapport_deforestation.pdf absent")
    with col3:
        p_gj = SHAPEFILES_DIR / "forets_tous_sites_toutes_annees.geojson"
        if p_gj.exists():
            with open(p_gj, "rb") as f:
                st.download_button("⬇️ GeoJSON multi-sites", f.read(),
                                   "forets_tous_sites.geojson", "application/json",
                                   use_container_width=True)
        else:
            st.caption("❌ GeoJSON multi-sites absent")

    st.divider()

    # --- Tableau statistiques détaillé (deforestation_stats.csv) ---
    st.subheader("Statistiques de déforestation détaillées (OS4)")
    csv_path = TABLES_DIR / "deforestation_stats.csv"
    if csv_path.exists():
        df_stats = pd.read_csv(csv_path)
        df_display = df_stats.rename(columns={
            "site": "Site", "year": "Année",
            "forest_ha": "Forêt (ha)", "deforest_ha": "Déforesté (ha)",
            "reforest_ha": "Reboisé (ha)", "net_change_ha": "Var. nette (ha)",
            "deforest_pct": "Taux déf. (%)"
        })
        def _style_row(row):
            if row["Année"] in [2015, 2024]:
                return ["background-color:#fff3cd"] * len(row)
            return [""] * len(row)
        st.dataframe(
            df_display.style.apply(_style_row, axis=1),
            hide_index=True, use_container_width=True
        )
        st.caption("⚠️ Lignes jaunes = années Sentinel-2 avec anomalie radiométrique (classification dégénérée).")
    else:
        st.info("deforestation_stats.csv introuvable.")


def _generate_pdf(type_rapport, periode, zone, graphiques, tableaux, methodo, recommandations) -> bytes:
    """Génère un rapport PDF avec ReportLab."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    vert   = colors.HexColor(VERT_FORET)
    rouge  = colors.HexColor(ROUGE_ALERTE)

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        textColor=vert, fontSize=18, spaceAfter=8, alignment=TA_CENTER
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        textColor=vert, fontSize=13, spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, spaceAfter=6, alignment=TA_JUSTIFY, leading=14
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, spaceAfter=4
    )

    story = []

    # --- Page de garde ---
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("🇹🇬 ForestGuard Togo", title_style))
    story.append(Paragraph(type_rapport, ParagraphStyle(
        "sub", parent=styles["Normal"], fontSize=13, textColor=rouge,
        spaceAfter=6, alignment=TA_CENTER
    )))
    story.append(HRFlowable(width="100%", thickness=2, color=vert))
    story.append(Spacer(1, 0.5*cm))

    meta_data = [
        ["Zone d'étude", zone],
        ["Période analysée", periode],
        ["Date de génération", datetime.now().strftime("%d/%m/%Y à %H:%M")],
        ["Modèle", "CNN 7 canaux (Accuracy=86.25%, AUC-ROC=1.0000)"],
        ["Sites", "Réserve d'Abdoulaye & Forêt de Fazao-Malfakassa"],
        ["Source données", "Sentinel-2 / Landsat (30 m, EPSG:32631)"],
    ]
    t = Table(meta_data, colWidths=[6*cm, 11*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(VERT_PALE)),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(
        "<i>Ce rapport a été généré automatiquement par l'application ForestGuard Togo, "
        "développée dans le cadre d'un mémoire de Master en Informatique (2026), "
        "Université de l'Afrique de l'Ouest (UCAO-UUT) — Département d'Informatique.</i>",
        caption_style
    ))
    story.append(PageBreak())

    # --- Résumé exécutif ---
    story.append(Paragraph("Résumé Exécutif", h2_style))
    kpis = DATA["kpis"]
    story.append(Paragraph(
        f"L'étude porte sur deux sites forestiers au Togo : la Réserve de Faune d'Abdoulaye "
        f"et la Forêt de Fazao-Malfakassa, couvrant la période {periode}. "
        f"La superficie forestière cumulée s'établissait à {kpis['superficie_initiale_ha']:,} ha "
        f"en début de période et atteint {kpis['superficie_valide_2020_ha']:,} ha en 2020 (dernières années valides Landsat), "
        f"soit une perte nette de {kpis['perte_totale_ha']:,} ha principalement sur le site de Fazao (2013-2020). "
        f"Le site de Fazao-Malfakassa présente le taux de déforestation le plus critique "
        f"({kpis['taux_deforestation_moyen_pct']}%/an en moyenne).",
        body_style
    ))

    # --- Données statistiques ---
    if tableaux:
        story.append(Paragraph("Données Statistiques par Site", h2_style))
        for site_name, site_data in DATA["sites"].items():
            if zone not in ("Toutes les zones", site_name):
                continue
            story.append(Paragraph(f"Site : {site_data['nom_complet']}", ParagraphStyle(
                "h3", parent=styles["Heading3"], fontSize=11, textColor=rouge
            )))
            sup_data = site_data["superficie_foret_ha"]
            header = ["Année", "Superficie (ha)", "Variation (ha)"]
            rows_t = [header]
            prev_val = None
            for yr, val in sup_data.items():
                variation = f"{val - prev_val:+,}" if prev_val is not None else "—"
                rows_t.append([yr, f"{val:,}", variation])
                prev_val = val
            t2 = Table(rows_t, colWidths=[4*cm, 7*cm, 6*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(VERT_FORET)),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f5faf5")]),
                ("PADDING",    (0, 0), (-1, -1), 5),
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.4*cm))

    # --- Méthodologie ---
    if methodo:
        story.append(PageBreak())
        story.append(Paragraph("Méthodologie", h2_style))
        for phase in DATA["pipeline_phases"]:
            story.append(Paragraph(
                f"<b>{phase['id']} — {phase['titre']}</b> | {phase['statut']}",
                ParagraphStyle("ph", parent=styles["Normal"],
                               fontSize=10, textColor=vert, spaceBefore=8)
            ))
            story.append(Paragraph(phase["description"], body_style))
            story.append(Paragraph(
                f"Outils : {', '.join(phase['outils'])}",
                caption_style
            ))

    # --- Recommandations ---
    if recommandations:
        story.append(PageBreak())
        story.append(Paragraph("Recommandations", h2_style))
        recs = [
            "Prioriser les actions de reforestation dans la Réserve d'Abdoulaye "
            "(taux de déforestation critique : >1.9%/an sur 2015-2024).",
            "Mettre en place un système de surveillance mensuelle par imagerie satellite "
            "Sentinel-2 (gratuit, 10 m de résolution) pour détecter les coupes précoces.",
            "Renforcer les patrouilles dans les zones à fort indice de suspicion "
            "identifiées par la heatmap NDVI du modèle CNN.",
            "Engager les communautés locales dans la gestion participative des forêts "
            "via des ateliers de sensibilisation.",
            "Étendre le modèle CNN à d'autres zones forestières du Togo "
            "(Kpalimé, Kloto, Tchaoudjo).",
        ]
        for i, rec in enumerate(recs, 1):
            story.append(Paragraph(f"{i}. {rec}", body_style))

    # --- Références APA ---
    story.append(PageBreak())
    story.append(Paragraph("Références", h2_style))
    for ref in DATA["references"]:
        story.append(Paragraph(f"• {ref}", caption_style))

    doc.build(story)
    return buf.getvalue()


# ============================================================
# ONGLET 5 — MÉTHODOLOGIE
# ============================================================
def tab_methodologie():
    st.header("ℹ️ Méthodologie — Pipeline de Recherche")
    st.markdown(
        "Présentation des 5 phases du pipeline méthodologique du mémoire de Master 2026, "
        "Université de l'Afrique de l'Ouest (UCAO-UUT) — Détection de la déforestation au Togo par Deep Learning."
    )

    # Schéma visuel du pipeline
    st.subheader("Schéma du Pipeline")
    if PLOTLY_OK:
        phases = DATA["pipeline_phases"]
        colors_map = {
            "✅": VERT_FORET,
            "⏳": ORANGE_WARN,
            "🔄": "#1f77b4",
        }
        fig_pipe = go.Figure()
        for i, p in enumerate(phases):
            c_key = next((k for k in colors_map if k in p["statut"]), "🔄")
            color = colors_map[c_key]
            fig_pipe.add_trace(go.Bar(
                x=[p["id"]], y=[1],
                marker_color=color,
                text=p["titre"], textposition="inside",
                name=p["id"],
                hovertext=p["statut"],
                showlegend=False,
                width=0.7,
            ))
        fig_pipe.update_layout(
            title="Pipeline méthodologique (OS1→OS5)",
            barmode="stack",
            height=180,
            margin=dict(t=40, b=20, l=10, r=10),
            plot_bgcolor=BLANC, paper_bgcolor=BLANC,
            yaxis_visible=False,
            xaxis_title="",
        )
        st.plotly_chart(fig_pipe, use_container_width=True)

    # Expanders détaillés par phase
    for phase in DATA["pipeline_phases"]:
        statut_icon = phase["statut"].split()[0]
        with st.expander(
            f"{statut_icon} **{phase['id']} — {phase['titre']}** | {phase['statut']}",
            expanded=False
        ):
            st.markdown(f"**Description :** {phase['description']}")
            st.markdown(f"**Outils :** {', '.join(f'`{o}`' for o in phase['outils'])}")

            if phase["id"] == "OS2":
                figs_os2 = [
                    ("temporal_indices.png",       "Évolution des indices spectraux (2000–2024)"),
                    ("index_maps_abdoulaye.png",    "Cartes d'indices spectraux — Abdoulaye"),
                    ("patch_7ch_example.png",       "Exemple de patch 7 canaux (256×256)"),
                ]
                for fname, cap in figs_os2:
                    p = FIGURES_DIR / fname
                    if p.exists():
                        st.image(str(p), caption=cap, use_container_width=True)

            if phase["id"] == "OS3":
                st.markdown("**Architecture CNN détaillée :**")
                st.code(
                    """
Conv2D(32, 3×3) + BatchNorm + MaxPool
Conv2D(64, 3×3) + BatchNorm + MaxPool
Conv2D(128, 3×3) + BatchNorm + MaxPool
Conv2D(256, 3×3) + BatchNorm + MaxPool
GlobalAveragePooling2D
Dense(256) + BatchNorm + Dropout(0.5)
Dense(64)  + BatchNorm + Dropout(0.3)
Dense(1, sigmoid)  → Forêt / Déforestation
                    """,
                    language="text"
                )
                m = DATA["modele"]["metriques"]
                st.markdown(
                    f"| Métrique | Valeur |\n|---|---|\n"
                    f"| Accuracy | **{m['accuracy']*100:.2f}%** |\n"
                    f"| AUC-ROC | **{m['auc_roc']:.4f}** |\n"
                    f"| Seuil optimal | **{DATA['modele']['seuil_optimal']}** |\n"
                    f"| Canaux | **{DATA['modele']['canaux']} (7-ch)** |\n"
                    f"| Patches total | **{m['nb_patches_total']:,}** (train={m['nb_train']}, val={m['nb_val']}, test={m['nb_test']}) |"
                )
                figs_os3 = [
                    ("training_curves_colab.png",   "Courbes d'entraînement (loss / accuracy)"),
                    ("confusion_matrix_colab.png",  "Matrice de confusion (test set)"),
                    ("roc_curve_colab.png",         "Courbe ROC (AUC=1.00)"),
                    ("sample_patches_by_class.png", "Exemples de patches par classe"),
                ]
                cols3 = st.columns(2)
                for i, (fname, cap) in enumerate(figs_os3):
                    p = FIGURES_DIR / fname
                    if p.exists():
                        with cols3[i % 2]:
                            st.image(str(p), caption=cap, use_container_width=True)

            if phase["id"] == "OS4":
                st.markdown("**Paramètres de fenêtre glissante :**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Taille patch", "256×256 px")
                    st.metric("Stride (pas)", "128 px (50% overlap)")
                with col_b:
                    st.metric("Abdoulaye", "~130 patches/image × 6 images")
                    st.metric("Fazao", "~400 patches/image × 4 images")
                figs_os4 = [
                    ("forest_area_timeline.png",    "Évolution de la superficie forestière"),
                    ("change_maps_abdoulaye.png",   "Cartes de changement — Abdoulaye"),
                    ("change_maps_fazao.png",       "Cartes de changement — Fazao"),
                ]
                for fname, cap in figs_os4:
                    p = FIGURES_DIR / fname
                    if p.exists():
                        st.image(str(p), caption=cap, use_container_width=True)

    st.divider()

    # Tableau des livrables OS5
    st.subheader("Livrables Opérationnels (OS5)")
    livrables = DATA["livrables_os5"]
    df_liv = pd.DataFrame(livrables).drop(columns=["id"])
    df_liv.columns = ["Livrable", "Format", "Statut"]

    statut_icons = {
        "planifié": "📋 Planifié",
        "en cours": "🔄 En cours",
        "complété": "✅ Complété",
        "entraînement Colab requis": "⏳ Colab GPU requis",
    }
    df_liv["Statut"] = df_liv["Statut"].map(lambda x: statut_icons.get(x, x))
    st.dataframe(df_liv, hide_index=True, use_container_width=True)

    # Références
    st.divider()
    st.subheader("Références principales")
    for ref in DATA["references"]:
        st.markdown(f"- {ref}")


# ============================================================
# APPLICATION PRINCIPALE
# ============================================================
def main():
    zone, threshold = render_sidebar()

    st.markdown(
        f"<h1 style='color:{VERT_FORET};margin-bottom:0;'>"
        " ForestGuard Togo</h1>"
        "<p style='color:#666;margin-top:4px;'>"
        "Système de détection de la déforestation par Deep Learning (CNN 7 canaux) — "
        "Réserve d'Abdoulaye &amp; Forêt de Fazao-Malfakassa | 2000–2024"
        "</p>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " 🔄Analyse d'Image",
        " ⏳Tableau de Bord",
        " ✅Cartographie OS5",
        "⏳ Comparaison Multi-Images",
        "📋 Rapports PDF",
        "ℹ️ Méthodologie",
    ])

    with tab1:
        tab_analyse(threshold, zone)
    with tab2:
        tab_dashboard(zone)
    with tab3:
        tab_cartographie(zone)
    with tab4:
        tab_comparaison(threshold)
    with tab5:
        tab_rapports()
    with tab6:
        tab_methodologie()


if __name__ == "__main__":
    main()
