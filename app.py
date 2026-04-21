import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG & ASSETS ---
MODEL_PATH = 'best_model_final.keras'
TASK_PATH = 'holistic_landmarker.task'
ENCODER_PATH = 'label_encoder.pkl'
NORM_MEAN = 'feature_mean.npy'
NORM_STD = 'feature_std.npy'

st.set_page_config(page_title="BITS Sign Language AI", layout="wide")
st.title("🤟 Real-Time ASL Recognition")
st.sidebar.header("System Settings")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    mean = np.load(NORM_MEAN)
    std = np.load(NORM_STD)
    
    # Initialize MediaPipe Tasks
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HolisticLandmarker.create_from_options(options)
    
    return model, le, mean, std, detector

try:
    model, le, mean, std, detector = load_all()
    st.sidebar.success("Model & Assets Loaded!")
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- UTILS ---
def extract_live_features(result):
    """ Corrected for MediaPipe Tasks API Landmark indexing """
    # Pose: result.pose_landmarks is a list of landmarks
    if result.pose_landmarks:
        # Note: We remove the [0] because result.pose_landmarks is the list itself
        pose = np.array([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks[0] if hasattr(result.pose_landmarks[0], '__iter__')] if isinstance(result.pose_landmarks[0], list) else [[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks]).flatten()
    else:
        pose = np.zeros(132)

    # Simplified and safe extraction:
    def get_coords(res_attr, size):
        if res_attr:
            # Tasks API returns a list of landmarks directly
            data = res_attr[0] if isinstance(res_attr[0], list) else res_attr
            return np.array([[l.x, l.y, l.z] for l in data]).flatten()
        return np.zeros(size)

    lh = get_coords(result.left_hand_landmarks, 63)
    rh = get_coords(result.right_hand_landmarks, 63)
    
    # Final check to ensure we always hit exactly 258
    full_feat = np.concatenate([pose, lh, rh])
    
    if len(full_feat) != 258:
        # If there's a slight mismatch, pad or trim to avoid model crash
        full_feat = np.pad(full_feat, (0, max(0, 258 - len(full_feat))))[:258]
        
    return full_feat

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.65)
reset_btn = st.sidebar.button("Reset Sequence")

with col1:
    st.subheader("Webcam Input")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

with col2:
    st.subheader("Analysis")
    prediction_bar = st.empty()
    status_text = st.empty()

# --- MAIN LOOP ---
if 'sequence' not in st.session_state or reset_btn:
    st.session_state.sequence = []

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Convert for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Task API Inference (Video Mode requires timestamp)
    frame_timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    # Feature Extraction
    feat = extract_live_features(result)
    st.session_state.sequence.append(feat)
    st.session_state.sequence = st.session_state.sequence[-30:] # Sliding window of 30
    
    # Model Prediction
    if len(st.session_state.sequence) == 30:
        # Prepare data (Normalize exactly like training)
        input_data = np.expand_dims(st.session_state.sequence, axis=0)
        input_data = (input_data - mean) / (std + 1e-8)
        
        prediction = model.predict(input_data, verbose=0)[0]
        top_idx = np.argsort(prediction)[-3:][::-1] # Top 3
        
        with prediction_bar.container():
            for i, idx in enumerate(top_idx):
                prob = prediction[idx]
                label = le.classes_[idx]
                
                # Visual logic: Only highlight if above threshold
                if i == 0 and prob > threshold:
                    st.markdown(f"### 🏆 {label.upper()}")
                
                st.write(f"{label}: {prob*100:.1f}%")
                st.progress(float(prob))
        
        status_text.info("Sequence full - Analyzing motion...")
    else:
        status_text.warning(f"Buffer: {len(st.session_state.sequence)}/30 frames")

    # Render frame
    FRAME_WINDOW.image(image_rgb)

cap.release()