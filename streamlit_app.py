import os
import tempfile

import streamlit as st
from PIL import Image

# Uncomment the next line if Streamlit Cloud runs out of memory.
# os.environ.setdefault("ONLY_MODEL", "wkaandemir")

from app.detector.model import AIDetector


st.set_page_config(
    page_title="AI Image Authenticity Detector",
    page_icon="🤖",
    layout="centered",
)


# --------------------------------------------------------------------
# Custom CSS for professional dark theme
# --------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #080b12;
        color: #f5f7fb;
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        text-align: center;
        margin-bottom: 0;
    }

    .main-title span {
        background: linear-gradient(90deg, #9d93ff, #65a8ff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #a6afc0;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .card {
        background: rgba(18, 24, 36, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }

    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        color: #9b91ff;
        font-weight: 800;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 5px;
    }

    .confidence-high {
        background: rgba(255, 77, 103, 0.12);
        color: #ff4d67;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        text-align: center;
    }

    .confidence-medium {
        background: rgba(255, 159, 67, 0.12);
        color: #ff9f43;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        text-align: center;
    }

    .confidence-low {
        background: rgba(245, 200, 76, 0.12);
        color: #f5c84c;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        text-align: center;
    }

    .model-name {
        font-weight: 700;
        color: #ffffff;
    }

    .model-prob {
        color: #9b91ff;
        font-weight: 800;
        float: right;
    }

    .swatch-container {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }

    .swatch {
        width: 50px;
        height: 50px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .disclaimer {
        color: #a6afc0;
        font-size: 13px;
        text-align: center;
        margin-top: 20px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load the ensemble detector once and cache it."""
    return AIDetector()


# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------
st.markdown('<div class="main-title">AI Image <span>Authenticity Detector</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Created by Syed Hissam Kazmi</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Please upload an image to begin.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception:
    st.error("Could not open the image. Please upload a valid image file.")
    st.stop()

st.image(image, use_container_width=True, caption="Uploaded Image")

if not st.button("🔍 Analyze Image", use_container_width=True):
    st.stop()

detector = load_detector()

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    image.save(tmp.name)
    tmp_path = tmp.name

with st.spinner("Analyzing image..."):
    result = detector.predict_detailed(tmp_path)

if result is None or result.get("ensemble") is None:
    st.error("AI detector is currently unavailable. Please try again later.")
    st.stop()

ai_prob = result["ensemble"]
human_prob = 1.0 - ai_prob
confidence = result.get("confidence")

# --------------------------------------------------------------------
# Main result card
# --------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="metric-label">AI Probability</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{ai_prob:.2%}</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-label">Human Probability</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{human_prob:.2%}</div>', unsafe_allow_html=True)

if confidence:
    if confidence == "HIGH":
        st.markdown('<div class="confidence-high">HIGH CONFIDENCE</div>', unsafe_allow_html=True)
    elif confidence == "MEDIUM":
        st.markdown('<div class="confidence-medium">MEDIUM CONFIDENCE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="confidence-low">LOW CONFIDENCE</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------
# Model predictions card
# --------------------------------------------------------------------
models = result.get("models", {})
if models:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Model Predictions</div>', unsafe_allow_html=True)

    for model_name, prob in models.items():
        pct = f"{prob:.2%}"
        st.markdown(
            f'<div><span class="model-name">{model_name}</span><span class="model-prob">{pct}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(float(prob), 1.0))

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------
# Signals and colours
# --------------------------------------------------------------------
signals = result.get("signals", [])
colors = result.get("dominant_colors", [])

if signals or colors:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if signals:
        st.markdown('<div class="metric-label">Detection Signals</div>', unsafe_allow_html=True)
        for s in signals:
            st.markdown(f"• {s}")

    if colors:
        st.markdown('<div class="metric-label" style="margin-top:15px;">Dominant Colours</div>', unsafe_allow_html=True)
        swatch_html = '<div class="swatch-container">'
        for c in colors:
            swatch_html += f'<div class="swatch" style="background-color:{c};" title="{c}"></div>'
        swatch_html += '</div>'
        st.markdown(swatch_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------
# Disclaimer
# --------------------------------------------------------------------
st.markdown('<div class="disclaimer">This result is probabilistic and should not be treated as definitive proof.</div>', unsafe_allow_html=True)