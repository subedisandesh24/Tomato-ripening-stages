# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ----------------------------
# Load YOLO models
# ----------------------------
fruit_model = YOLO("fruit.pt")          # Detection model
leaf_model = YOLO("leafdisease.pt")     # Classification model

# ----------------------------
# Helper functions
# ----------------------------
def annotate_image(results, conf_thres=0.6):
    """Annotate detection results with colored bounding boxes."""
    img = results[0].orig_img.copy()
    for box in results[0].boxes:
        if box.conf < conf_thres:
            continue
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 0, 255) if label.lower() == "red" else (0, 255, 0) if label.lower() == "green" else (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

def count_fruits(results, conf_thres=0.6):
    """Count fruits per stage."""
    counts = {"Red":0, "Turning":0, "Green":0}
    for box in results[0].boxes:
        if box.conf < conf_thres:
            continue
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id].capitalize()
        if label in counts:
            counts[label] += 1
    return counts

def estimate_weight(length_cm, width_cm, density=900):
    """Estimate fruit weight using geometric approximation."""
    if abs(length_cm - width_cm) < 0.1 * length_cm:  # Sphere approx
        r = width_cm / 2 / 100  # convert cm to m
        V = (4/3) * np.pi * (r**3)
    else:  # Ellipsoid approx
        L = length_cm / 100
        W = width_cm / 100
        V = (4/3) * np.pi * (L/2) * (W/2)**2
    M = V * density  # kg
    return M

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🍅 Tomato Monitoring App")
tab1, tab2, tab3, tab4 = st.tabs(["Fruit Detector", "Video Mode", "Leaf Disease Classifier", "Fruit Weight Estimation"])

# ----------------------------
# Tab 1: Fruit Detector
# ----------------------------
with tab1:
    st.header("Fruit Detector")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","gif","bmp","tiff","webp","heif","heic","svg","eps","raw","psd"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        results = fruit_model.predict(img_np, conf=0.6)
        annotated = annotate_image(results, conf_thres=0.6)
        counts = count_fruits(results, conf_thres=0.6)

        # Show table
        st.subheader("Detection Summary")
        total = sum(counts.values())
        st.table({"Stage": list(counts.keys())+["Total"], "Count": list(counts.values())+[total]})

        # Show annotated image
        st.image(annotated, caption="Annotated Detection", use_column_width=True)

        # Download option
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmpfile.name, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        with open(tmpfile.name, "rb") as f:
            st.download_button("Download Annotated Image", f, file_name="detected.jpg")

# ----------------------------
# Tab 2: Video Mode
# ----------------------------
with tab2:
    st.header("Video Mode")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4","avi","mov","wmv","flv","mkv","webm","mpeg","mpg","3gp","avchd"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = fruit_model.predict(frame, conf=0.6)
            annotated = annotate_image(results, conf_thres=0.6)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()

        with open(out_path, "rb") as f:
            st.download_button("Download Annotated Video", f, file_name="detected_video.mp4")

# ----------------------------
# Tab 3: Leaf Disease Classifier
# ----------------------------
with tab3:
    st.header("Leaf Disease Classifier")
    leaf_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png","bmp","tiff","webp"])
    if leaf_file:
        image = Image.open(leaf_file).convert("RGB")
        img_np = np.array(image)
        results = leaf_model.predict(img_np)
        probs = results[0].probs
        if probs is not None:
            classes = results[0].names
            top3_idx = probs.topk(3).indices.cpu().numpy()
            st.subheader("Top 3 Predictions")
            for idx in top3_idx:
                st.write(f"{classes[idx]}: {probs[idx]:.2f}")
            st.write("Recommendation Strategy: (to be added)")

# ----------------------------
# Tab 4: Fruit Weight Estimation
# ----------------------------
with tab4:
    st.header("Fruit Weight Estimation")
    fruit_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","tiff","webp"])
    if fruit_file:
        image = Image.open(fruit_file).convert("RGB")
        img_np = np.array(image)
        results = fruit_model.predict(img_np, conf=0.6)

        # Calibration input
        st.info("Click two points on the image to set scale (not implemented in Streamlit yet).")
        scale_cm = st.number_input("Enter real-world length (cm) between two points:", min_value=0.1, value=5.0)

        # For demo: assume 1 pixel = 0.1 cm
        px_to_cm = 0.1

        counts = count_fruits(results, conf_thres=0.6)
        total = sum(counts.values())
        weights = []

        annotated = results[0].orig_img.copy()
        for box in results[0].boxes:
            if box.conf < 0.6:
                continue
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id].capitalize()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            length_px = y2 - y1
            width_px = x2 - x1
            length_cm = length_px * px_to_cm
            width_cm = width_px * px_to_cm
            weight = estimate_weight(length_cm, width_cm)
            weights.append(weight)

            color = (0, 0, 255) if label == "Red" else (0, 255, 0) if label == "Green" else (0, 255, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, f"{label} {weight:.3f} kg", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.image(annotated, caption="Annotated with weights", use_column_width=True)

        # Download option
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmpfile.name, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        with open(tmpfile.name, "rb") as f:
            st.download_button("Download Annotated Image", f, file_name="weights.jpg")

              # Table summary
        st.subheader("Yield Summary")
        total_weight = sum(weights)
        summary = {
            "Stage": list(counts.keys()) + ["Total"],
            "Count": list(counts.values()) + [total],
            "Weight (kg)": [round(w, 3) for w in weights] + [round(total_weight, 3)]
        }
        st.table(summary)

