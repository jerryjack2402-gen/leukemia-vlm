import streamlit as st
from PIL import Image
import numpy as np

from inference import VLMInferencePipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Leukemia VLM Diagnostic System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Pipeline ---
@st.cache_resource
def load_pipeline():
    # Cache to avoid reloading the ~500MB+ models constantly
    return VLMInferencePipeline()

pipeline = load_pipeline()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .report-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e86c1;
        font-family: 'Courier New', Courier, monospace;
        color: #333;
    }
    .metric-healthy { color: #28b463; font-weight: bold; font-size: 24px; }
    .metric-leukemia { color: #e74c3c; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063166.png", width=50) # Generic medical icon
    st.title("Automated Blood Smear Analysis")
    st.write("""
    **Vision Large Language Model Framework**
    
    This tool integrates a Vision Transformer (ViT) with a Medical LLM to provide:
    - 🎯 Automated Classification
    - 📝 Structured Pathology Reporting
    - 🔦 Diagnostic Explainability (XAI)
    """)
    st.markdown("---")
    st.caption("For Research Purposes Only. Not intended for direct clinical diagnosis without physician oversight.")

# --- Main App ---
st.title("Microscopic Image Analysis")
st.write("Upload a peripheral blood smear image to generate a diagnostic report.")

uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Input")
        st.image(image, use_container_width=True, caption="Microscopic Blood Smear")
        
    with st.spinner("VLM architecture is analyzing morphology and generating report..."):
        # Run inference
        results = pipeline.predict(image)
        
    with col2:
        st.subheader("Explainable AI (XAI) Heatmap")
        st.image(results['heatmap'], use_container_width=True, caption="Vision Transformer Attention Rollout")
        
    st.markdown("---")
    
    # Results Section
    st.header("Diagnostic Results")
    
    pred_class = results['prediction']
    conf_score = results['confidence'] * 100
    
    metric_class = "metric-leukemia" if pred_class == "Leukemia" else "metric-healthy"
    
    st.markdown(f"**Classification:** <span class='{metric_class}'>{pred_class}</span> (Confidence: {conf_score:.1f}%)", unsafe_allow_html=True)
    
    st.subheader("Automated Pathology Report")
    st.markdown(f"<div class='report-box'>{results['report']}</div>", unsafe_allow_html=True)

    # Allow downloading report
    st.download_button(
        label="Download Report as TXT",
        data=f"VLM Diagnostics Report\nClassification: {pred_class}\nConfidence: {conf_score:.1f}%\n\nNote:\n{results['report']}",
        file_name=f"pathology_report_{pred_class.lower()}.txt",
        mime="text/plain"
    )
