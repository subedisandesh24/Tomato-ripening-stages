import sys, streamlit as st
st.write(f"Python version: {sys.version}")
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pillow_heif
import cv2
import io
import tempfile
import os

st.title("Tomato Ripening Stage Detector 🍅")

# Load YOLO model (make sure fruit.pt is in the same folder as app.py)
model_path = os.path.join(os.path.dirname(__file__), "fruit.pt")
model = YOLO(model_path)

# Create tabs
tab1, tab2 = st.tabs(["🖼️ Image Mode", "📹 Video Mode"])

# ---------------- IMAGE MODE ----------------
with tab1:
    uploaded = st.file_uploader("Upload a tomato image", type=["jpg", "png", "jpeg", "heic"])
    if uploaded:
        # Handle HEIC images
        if uploaded.type == "image/heic":
            heif_file = pillow_heif.read_heif(uploaded.read())
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            img = Image.open(uploaded)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Run YOLO detection
        results = model(img)
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detections", use_column_width=True)

        # Download button
        result_pil = Image.fromarray(result_img)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("Download Detection Result", buf.getvalue(),
                           file_name="tomato_detection.png", mime="image/png")

        # Count tomatoes by ripening stage
        counts = {"Red": 0, "Green": 0, "Turning": 0}
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if "red" in label.lower():
                counts["Red"] += 1
            elif "green" in label.lower():
                counts["Green"] += 1
            elif "turning" in label.lower():
                counts["Turning"] += 1

        st.subheader("Tomato Counts by Stage")
        st.write(counts)

# ---------------- VIDEO MODE ----------------
with tab2:
    uploaded_video = st.file_uploader("Upload a tomato video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            result_frame = results[0].plot()
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

            stframe.image(result_frame, channels="RGB", use_column_width=True)

        cap.release()
