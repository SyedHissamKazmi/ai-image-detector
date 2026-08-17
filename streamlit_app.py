from __future__ import annotations

import hashlib
import html
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from app.detector.model import AIDetector

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AI Image Authenticity Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS (only for background, font, and minor overrides)
# ============================================================================
GLOBAL_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #080b12 0%, #0f1419 50%, #0a0d14 100%);
        color: #f5f7fb;
        font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .css-1d391kg {
        background-color: rgba(18, 24, 36, 0.7);
    }
    .stMetric {
        background: rgba(18, 24, 36, 0.8);
        border: 1px solid rgba(157, 147, 255, 0.2);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #65a8ff, #9d93ff);
    }
    .stButton > button {
        background: linear-gradient(90deg, #9d93ff, #65a8ff);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #7c6cff, #4d8cff);
        box-shadow: 0 15px 30px rgba(157, 147, 255, 0.3);
        transform: translateY(-2px);
    }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts for session history
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None  # full result from detector
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None  # hash of uploaded file
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.5  # adjustable threshold

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_detector():
    """Load the ensemble detector once and cache it."""
    return AIDetector()


def get_confidence_level(prob: float) -> Tuple[str, str]:
    """Return a human-readable confidence level and a color."""
    if prob >= 0.8:
        return "HIGH CONFIDENCE", "#ff4d67"
    elif prob >= 0.4:
        return "MEDIUM CONFIDENCE", "#ff9f43"
    else:
        return "LOW CONFIDENCE", "#4caf50"


def get_gauge_color(prob: float) -> str:
    """Return a color for the gauge based on probability."""
    if prob <= 0.4:
        return "#4caf50"
    elif prob <= 0.6:
        return "#ffc107"
    elif prob <= 0.8:
        return "#ff9f43"
    else:
        return "#ff4d67"


def create_gauge_chart(prob: float) -> go.Figure:
    """
    Create an animated gauge chart using Plotly.

    This replaces the previous custom SVG gauge entirely, eliminating
    any raw HTML rendering issues.
    """
    color = get_gauge_color(prob)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 48, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#a6afc0"},
                "bar": {"color": color},
                "bgcolor": "rgba(255, 255, 255, 0.05)",
                "borderwidth": 2,
                "bordercolor": "rgba(157, 147, 255, 0.3)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(76, 175, 80, 0.15)"},
                    {"range": [40, 60], "color": "rgba(255, 193, 7, 0.15)"},
                    {"range": [60, 80], "color": "rgba(255, 159, 67, 0.15)"},
                    {"range": [80, 100], "color": "rgba(255, 77, 103, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a6afc0"},
    )
    return fig


def extract_dominant_colors(image: Image.Image, count: int = 5) -> List[str]:
    """
    Extract dominant colors from the image using PIL.

    Returns a list of hex color strings.
    """
    try:
        sample = image.copy()
        sample.thumbnail((200, 200), Image.Resampling.LANCZOS)
        sample = sample.convert("RGB")
        sample = sample.quantize(colors=max(count, 5), method=Image.Quantize.MEDIANCUT)
        palette = sample.getpalette()
        color_counts = sample.getcolors()

        if not palette or not color_counts:
            return []

        color_counts.sort(reverse=True)
        colors = []
        for _, palette_index in color_counts[:count]:
            base = palette_index * 3
            if base + 2 >= len(palette):
                continue
            r, g, b = palette[base], palette[base + 1], palette[base + 2]
            colors.append(f"#{r:02X}{g:02X}{b:02X}")
        return colors
    except Exception:
        return []


def get_detection_signals(ai_prob: float, models: Dict[str, float]) -> List[Tuple[str, str]]:
    """
    Generate human-readable signals about the detection result.

    Returns a list of (signal_text, emoji) tuples.
    """
    signals = []

    # Ensemble strength signal
    if ai_prob >= 0.8:
        signals.append(("ML ensemble: strong AI signal", "🔴"))
    elif ai_prob <= 0.2:
        signals.append(("ML ensemble: strong human signal", "🟢"))
    elif 0.4 < ai_prob < 0.6:
        signals.append(("ML ensemble: uncertain", "🟡"))

    # Model agreement
    valid_models = {k: v for k, v in models.items() if v is not None}
    if len(valid_models) >= 2:
        probs = list(valid_models.values())
        diff = max(probs) - min(probs)
        if diff < 0.1:
            signals.append(("Models strongly agree", "✅"))
        elif diff > 0.3:
            signals.append(("Models show significant variance", "⚠️"))

    return signals


def get_model_agreement(models: Dict[str, float]) -> Tuple[str, str]:
    """
    Analyze model agreement.

    Returns (agreement_text, emoji).
    """
    valid_models = {k: v for k, v in models.items() if v is not None}
    if len(valid_models) < 2:
        return "Insufficient models for agreement check", "❓"

    probs = list(valid_models.values())
    avg = sum(probs) / len(probs)
    diff = max(probs) - min(probs)

    if diff < 0.1:
        if avg >= 0.8:
            return f"✅ Strong Agreement: Both models detect AI ({avg:.0%})", "🎯"
        elif avg <= 0.2:
            return f"✅ Strong Agreement: Both models detect Human ({(1-avg):.0%})", "🎯"
        else:
            return f"❓ Uncertain Agreement: Models uncertain ({avg:.0%})", "❓"
    else:
        return f"⚠️ Disagreement: Models differ by {diff:.0%}", "⚔️"


def hash_file(file_bytes: bytes) -> str:
    """Return an MD5 hash of file bytes."""
    return hashlib.md5(file_bytes).hexdigest()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("🧠 About")
    st.write("""
    This app uses two AI models to estimate whether an image is AI-generated.
    """)

    st.markdown("---")

    st.subheader("⚙️ Settings")
    new_threshold = st.slider(
        "Confidence threshold for classification",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Adjust the threshold used to decide between AI and human.",
    )
    if new_threshold != st.session_state.confidence_threshold:
        st.session_state.confidence_threshold = new_threshold

    st.markdown("---")

    st.subheader("🤖 Models")
    st.write("- Ateeqq/ai-vs-human-image-detector")
    st.write("- wkaandemir/ai-image-detector")

    st.markdown("---")

    st.subheader("👤 Created by")
    st.write("Syed Hissam Kazmi")

    st.markdown("---")

    st.subheader("🔗 Links")
    st.write("[GitHub Repository](https://github.com/SyedHissamKazmi/ai-image-detector)")
    st.write("[Live Demo](https://syedhissamkazmi-ai-image-detector.streamlit.app)")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("AI Image Authenticity Detector")
st.caption("Detect AI-generated vs. authentic images with precision")
st.markdown("---")

# ============================================================================
# FILE UPLOAD
# ============================================================================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("👆 Upload an image to begin analysis")
    st.stop()

# ============================================================================
# DETECT NEW FILE
# ============================================================================
file_bytes = uploaded_file.getvalue()
file_hash = hash_file(file_bytes)

if file_hash != st.session_state.current_file_hash:
    # New file uploaded, reset analysis
    st.session_state.current_file_hash = file_hash
    st.session_state.analysis_result = None

# ============================================================================
# IMAGE PREVIEW & METADATA
# ============================================================================
try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"❌ Error loading image: {e}")
    st.stop()

col_img, col_meta = st.columns([2, 1], gap="large")

with col_img:
    st.image(image, use_container_width=True, caption="Uploaded Image")

with col_meta:
    st.subheader("📋 Image Metadata")
    filename_escaped = html.escape(uploaded_file.name)
    image_format = image.format or (
        uploaded_file.name.rsplit('.', 1)[-1].upper()
        if '.' in uploaded_file.name else 'Unknown'
    )
    st.write(f"**Filename:** {filename_escaped}")
    st.write(f"**Format:** {image_format}")
    st.write(f"**Dimensions:** {image.width} × {image.height} px")
    st.write(f"**File Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")

# ============================================================================
# ANALYZE BUTTON
# ============================================================================
st.markdown("---")
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    analyze_clicked = st.button("🔍 Analyze Image", use_container_width=True)

if analyze_clicked:
    detector = load_detector()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, quality=95)
        tmp_path = tmp.name

    with st.spinner("🔬 Analyzing with AI detector ensemble..."):
        result = detector.predict_detailed(tmp_path)

    if result is None or result.get("ensemble") is None:
        st.error("❌ Detector Error: The AI detector is currently unavailable.")
        st.stop()

    # Store result in session state
    st.session_state.analysis_result = result

    # Extract derived data
    ai_prob = result["ensemble"]
    human_prob = 1.0 - ai_prob
    confidence_text, confidence_color = get_confidence_level(ai_prob)
    models = result.get("models", {})
    signals = get_detection_signals(ai_prob, models)
    colors = extract_dominant_colors(image)
    agreement_text, agreement_emoji = get_model_agreement(models)

    # Append to session history
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "filename": uploaded_file.name,
        "ai_probability": f"{ai_prob:.1%}",
        "human_probability": f"{human_prob:.1%}",
        "confidence": confidence_text,
    })

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
if st.session_state.analysis_result is None:
    st.info("👆 Click the button above to analyze this image")
    st.stop()

result = st.session_state.analysis_result
ai_prob = result["ensemble"]
human_prob = 1.0 - ai_prob
confidence_text, confidence_color = get_confidence_level(ai_prob)
models = result.get("models", {})
signals = get_detection_signals(ai_prob, models)
colors = extract_dominant_colors(image)
agreement_text, agreement_emoji = get_model_agreement(models)

st.markdown("---")
st.subheader("📊 Analysis Results")

# ============================================================================
# GAUGE & METRICS
# ============================================================================
col_gauge, col_metrics = st.columns([1, 1], gap="large")

with col_gauge:
    gauge_fig = create_gauge_chart(ai_prob)
    st.plotly_chart(gauge_fig, use_container_width=True)

with col_metrics:
    st.subheader("Key Metrics")
    col_ai, col_human = st.columns(2)
    with col_ai:
        st.metric("AI Probability", f"{ai_prob:.1%}")
    with col_human:
        st.metric("Human Probability", f"{human_prob:.1%}")

    # Confidence badge using st.markdown with colored HTML (safe, no card)
    st.markdown(
        f"""<div style="
            display: inline-block;
            padding: 8px 16px;
            border-radius: 30px;
            background-color: {confidence_color};
            color: white;
            font-weight: bold;
            margin-top: 20px;
        ">{confidence_text}</div>""",
        unsafe_allow_html=True,
    )

# ============================================================================
# MODEL AGREEMENT
# ============================================================================
st.markdown(
    f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 20px;
        background-color: rgba(18, 24, 36, 0.8);
        border: 1px solid rgba(157, 147, 255, 0.2);
        border-radius: 15px;
        margin: 20px 0;
    ">
        <span style="font-size: 2rem;">{agreement_emoji}</span>
        <span style="color: #f5f7fb; font-weight: 600;">{agreement_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================
st.subheader("🤖 Model Predictions")

if models:
    cols = st.columns(len(models))
    for idx, (model_name, prob) in enumerate(models.items()):
        with cols[idx]:
            if prob is not None:
                st.write(f"**{model_name.title()}**")
                st.progress(min(float(prob), 1.0))
                st.write(f"{prob:.1%}")
                if prob >= 0.8:
                    st.write("⬆ Strong AI Signal")
                elif prob <= 0.2:
                    st.write("⬇ Strong Human Signal")
                else:
                    st.write("↔ Uncertain")
            else:
                st.write(f"**{model_name.title()}**")
                st.write("Model unavailable")
else:
    st.info("No model predictions available.")

# ============================================================================
# DETECTION SIGNALS & DOMINANT COLORS
# ============================================================================
col_signals, col_colors = st.columns([1, 1], gap="large")

with col_signals:
    if signals:
        st.subheader("⚡ Detection Signals")
        for signal_text, emoji in signals:
            st.write(f"{emoji} {signal_text}")

with col_colors:
    if colors:
        st.subheader("🎨 Dominant Colors")
        # Use st.columns to display swatches
        color_cols = st.columns(len(colors))
        for color_hex, color_col in zip(colors, color_cols):
            with color_col:
                st.markdown(
                    f"""<div style="
                        width: 80px;
                        height: 80px;
                        background-color: {color_hex};
                        border-radius: 10px;
                        border: 1px solid rgba(255,255,255,0.2);
                    "></div>
                    <small>{color_hex}</small>""",
                    unsafe_allow_html=True,
                )

# ============================================================================
# DETAILED METADATA EXPANDER
# ============================================================================
with st.expander("📋 Detailed Analysis Metadata", expanded=False):
    st.write(f"**Analysis Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"**Ensemble Result:** AI Probability: {ai_prob:.6f}")
    st.write("**Individual Model Results:**")
    for name, prob in models.items():
        if prob is not None:
            st.write(f"- {name}: {prob:.6f}")
    st.write(f"**Detected Colors:** {', '.join(colors) if colors else 'No colors detected'}")

# ============================================================================
# SESSION HISTORY
# ============================================================================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Session History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7d8698; font-size: 0.9rem; padding: 20px;">
        <p><strong>⚠️ Disclaimer:</strong> This result is probabilistic and should not be treated as definitive proof. 
        Use this tool as one indicator among multiple verification methods.</p>
        <p>Powered by ensemble of Ateeqq/ai-vs-human-image-detector & wkaandemir/ai-image-detector</p>
    </div>
    """,
    unsafe_allow_html=True,
)