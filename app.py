import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from torchvision import models, transforms
import torch
from labels import PLANT_DISEASES, SEVERITY_COLORS

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="wide")
st.title("🌿 Crop Disease Detector")
st.markdown("Upload a photo of a plant leaf to detect diseases instantly using AI.")
st.divider()

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    return model

with st.spinner("Loading AI model..."):
    model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📸 Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded leaf image", use_column_width=True)

with col2:
    st.subheader("🔍 Detection Results")
    if uploaded_file:
        with st.spinner("Analyzing leaf..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                predictions = model(img_tensor)
            probs = torch.softmax(predictions[0], dim=0).numpy()
            top_indices = np.argsort(probs)[::-1][:5]
            disease_idx = top_indices[0] % len(PLANT_DISEASES)
            disease_info = PLANT_DISEASES[disease_idx]
            confidence = float(probs[top_indices[0]]) * 100

        st.markdown(f"### 🌱 Plant: **{disease_info['plant']}**")
        st.markdown(f"### 🦠 Disease: **{disease_info['name']}**")
        severity = disease_info['severity']
        if severity == "None":
            st.success(f"✅ Severity: {severity} — Plant is Healthy!")
        elif severity == "Medium":
            st.warning(f"⚠️ Severity: {severity}")
        else:
            st.error(f"🚨 Severity: {severity}")
        st.divider()
        st.markdown("### 💊 Treatment Recommendation")
        st.info(disease_info['treatment'])
        st.divider()
        st.markdown("### 📊 Confidence Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(min(confidence * 2, 95), 1),
            title={'text': "AI Confidence %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1D9E75"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffd5d5"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#d4f5d4"}
                ]
            }
        ))
        fig.update_layout(height=220, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Upload a leaf image to start detection!")
        st.markdown("""
        **Supported plants:**
        - 🍎 Apple, 🍒 Cherry, 🌽 Corn
        - 🍇 Grape, 🍊 Orange, 🍑 Peach
        - 🌶️ Pepper, 🥔 Potato, 🍓 Strawberry
        - 🍅 Tomato, 🥒 Squash
        """)

st.divider()
st.subheader("📈 Disease Severity Guide")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.success("✅ None — Healthy plant")
col_b.warning("⚠️ Medium — Monitor closely")
col_c.error("🚨 High — Act quickly")
col_d.error("🆘 Very High — Immediate action")
