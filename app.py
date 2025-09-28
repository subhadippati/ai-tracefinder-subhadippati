import streamlit as st
import joblib
import cv2
import numpy as np
import pandas as pd

# Load your saved model and label encoder
MODEL_FILE = "rf_model.pkl"
ENCODER_FILE = "label_encoder.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    le = joblib.load(ENCODER_FILE)
    return model, le

model, le = load_model()

IMG_SIZE = 128

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def extract_fft_features(gray_image):
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum.flatten()

def extract_features(image):
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    gray = convert_to_grayscale(image_resized)
    pixels = gray.flatten()
    fft_features = extract_fft_features(gray)
    return np.hstack([pixels, fft_features])

# Streamlit UI
st.title("📄 TraceFinder – Scanner Identification")

uploaded_files = st.file_uploader("Upload scanned images", accept_multiple_files=True, type=["jpg","jpeg","png","tif"])

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            feat = extract_features(img).reshape(1, -1)
            pred = model.predict(feat)
            probs = model.predict_proba(feat)
            conf = np.max(probs) * 100
            scanner_name = le.inverse_transform(pred)[0]
            results.append([uploaded_file.name, scanner_name, conf])

    df = pd.DataFrame(results, columns=["File", "Predicted Scanner", "Confidence"])
    st.table(df)
    df.to_csv("predictions.csv", index=False)
    st.success("📑 Predictions saved to predictions.csv")
