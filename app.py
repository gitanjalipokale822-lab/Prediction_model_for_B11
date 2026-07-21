import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from streamlit_lottie import st_lottie

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS (Glassmorphism UI)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Neural AI Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Animated CSS styling
st.markdown("""
    <style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Card Glassmorphism effect */
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Custom Header Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0px;
    }
    
    /* Pulse button animation */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        font-weight: 600;
        border: none;
        padding: 12px 28px;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
        background: linear-gradient(90deg, #4f46e5, #9333ea);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load Lottie animations via URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a neural network animation
lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m64ro19g.json")

# ---------------------------------------------------------
# 2. Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_keras_model():
    # Load your pickle/keras model safely
    return tf.keras.models.load_model("model.pkl")

try:
    model = load_keras_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Sidebar Setup
# ---------------------------------------------------------
with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=200, key="ai_anim")
    st.title("⚙️ Model Settings")
    st.write("This app runs inference using a 4-feature Sequential Neural Network[cite: 1].")
    
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.caption("🚀 Powered by Streamlit & TensorFlow")

# ---------------------------------------------------------
# 4. Main Interface
# ---------------------------------------------------------
st.markdown('<p class="main-title">🧠 Neural Model Predictor</p>', unsafe_allow_html=True)
st.write("Adjust the input parameters below to generate predictions from the deep learning model.")

st.markdown("<br>", unsafe_allow_html=True)

# Input parameters split into columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('### 📊 Primary Features')
    feature_1 = st.number_input("Feature 1", value=0.0, step=0.1, help="Enter numerical value for Input 1")
    feature_2 = st.number_input("Feature 2", value=0.0, step=0.1, help="Enter numerical value for Input 2")

with col2:
    st.markdown('### 📈 Secondary Features')
    feature_3 = st.number_input("Feature 3", value=0.0, step=0.1, help="Enter numerical value for Input 3")
    feature_4 = st.number_input("Feature 4", value=0.0, step=0.1, help="Enter numerical value for Input 4")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Prediction Logic & Display
# ---------------------------------------------------------
if st.button("✨ Run Prediction"):
    # Format inputs into shape (1, 4) for model inference
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]], dtype=np.float32)
    
    with st.spinner("Analyzing inputs through neural layers..."):
        prediction_prob = float(model.predict(input_data)[0][0])
        predicted_class = 1 if prediction_prob >= threshold else 0

    st.markdown("---")
    st.subheader("🎯 Output Analysis")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="Confidence Probability", 
            value=f"{prediction_prob * 100:.2f}%", 
            delta=f"{'Above' if prediction_prob >= threshold else 'Below'} Threshold"
        )
        st.progress(prediction_prob)

    with res_col2:
        if predicted_class == 1:
            st.success(f"**Predicted Class: 1** (Probability $\ge$ {threshold})")
            st.balloons()
        else:
            st.info(f"**Predicted Class: 0** (Probability < {threshold})")
