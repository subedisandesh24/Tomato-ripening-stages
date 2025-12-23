import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pillow_heif
import cv2
import io
import tempfile
import os
import numpy as np

st.title("Tomato Monitoring System 🍅🌿")

# ---------------- Load Models ----------------
fruit_model = YOLO(os.path.join(os.path.dirname(__file__), "fruit.pt"))
disease_model = YOLO(os.path.join(os.path.dirname(__file__), "leafdisease.pt"))

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs([
    "🖼️ Fruit Image Detector", 
    "📹 Fruit Video Detector", 
    "🦠 Leaf Disease Classifier"
])

# ---------------- FRUIT IMAGE DETECTOR ----------------
with tab1:
    uploaded = st.file_uploader("Upload a tomato image", type=["jpg", "png", "jpeg", "heic"])
    if uploaded:
        if uploaded.type == "image/heic":
            heif_file = pillow_heif.read_heif(uploaded.read())
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            img = Image.open(uploaded)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        results = fruit_model(img)
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
            label = fruit_model.names[cls]
            if "red" in label.lower():
                counts["Red"] += 1
            elif "green" in label.lower():
                counts["Green"] += 1
            elif "turning" in label.lower():
                counts["Turning"] += 1

        st.subheader("Tomato Counts by Stage")
        st.write(counts)

# ---------------- FRUIT VIDEO DETECTOR ----------------
with tab2:
    uploaded_video = st.file_uploader("Upload a tomato video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        input_path = tfile.name
        output_path = input_path.replace(".mp4", "_detected.mp4")

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = fruit_model(frame)
            result_frame = results[0].plot()

            out.write(result_frame)

            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            stframe.image(result_frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
        out.release()

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Detected Video",
                data=f.read(),
                file_name="tomato_detected.mp4",
                mime="video/mp4"
            )

# ---------------- LEAF DISEASE CLASSIFIER ----------------
with tab3:
    disease_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "png", "jpeg", "heic"])
    if disease_file:
        if disease_file.type == "image/heic":
            heif_file = pillow_heif.read_heif(disease_file.read())
            disease_img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            disease_img = Image.open(disease_file)

        st.image(disease_img, caption="Uploaded Leaf Image", use_column_width=True)

        disease_results = disease_model(disease_img)
        probs = disease_results[0].probs

        # Top-3 predictions
        top3_indices = probs.top5[:3]
        st.subheader("Top-3 Disease Predictions 🦠")
        for idx in top3_indices:
            class_name = disease_model.names[idx]
            confidence = probs.data[idx]
            st.write(f"- {class_name}: {confidence:.2f}")

        # Highlight the major disease (top-1)
        major_idx = probs.top1
        major_class = disease_model.names[major_idx]
        major_conf = probs.top1conf

        # Convert PIL to OpenCV
        cv_img = cv2.cvtColor(np.array(disease_img), cv2.COLOR_RGB2BGR)

        # Draw a red circle in the center of the image
        h, w, _ = cv_img.shape
        center = (w // 2, h // 2)
        radius = min(h, w) // 4
        cv2.circle(cv_img, center, radius, (0, 0, 255), 5)

        # Put disease label text
        cv2.putText(cv_img, f"{major_class} ({major_conf:.2f})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Convert back to RGB for Streamlit
        result_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption=f"Major Disease Highlighted: {major_class}", use_column_width=True)
