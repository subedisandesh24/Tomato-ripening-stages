app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Tomato Ripening Stage Detector 🍅")

# Load your model
model = YOLO("best.pt")

# Upload image
uploaded = st.file_uploader("Upload a tomato image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(img)
    result_img = results[0].plot()

    st.image(result_img, caption="Detections")

    # Count tomatoes by class
    counts = {}
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        counts[label] = counts.get(label, 0) + 1

    st.subheader("Tomato Counts")
    st.write(counts)
