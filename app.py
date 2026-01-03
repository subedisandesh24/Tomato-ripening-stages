import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pillow_heif
import cv2
import io
import tempfile
import os
import numpy as np
import re

st.set_page_config(page_title="Tomato Monitoring System", layout="wide")
st.title("Tomato Monitoring System 🍅🌿")


# Load models
fruit_model = YOLO("fruit.pt")
disease_model = YOLO("leafdisease.pt")

# Define tabs
tab1, tab2, tab3 = st.tabs([
    "🖼️ Fruit Image Detector", 
    "📹 Fruit Video Detector", 
    "🦠 Leaf Disease Classifier"
])

# Fruit Image Detector
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

        result_pil = Image.fromarray(result_img)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("Download Detection Result", buf.getvalue(),
                           file_name="tomato_detection.png", mime="image/png")

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

# Fruit Video Detector
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

# Leaf Disease Classifier
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

        top3_indices = probs.top5[:3]
        st.subheader("Top-3 Disease Predictions 🌿")
        for idx in top3_indices:
            class_name = disease_model.names[idx]
            confidence = probs.data[idx]
            if idx == probs.top1:
                st.markdown(f"- 🔴 **{class_name}** → Confidence: `{confidence:.2f}`")
            else:
                st.write(f"- {class_name}: {confidence:.2f}")

        # Normalize top-1 class name
        raw_class = disease_model.names[probs.top1]
        major_class = re.sub(r'[^a-zA-Z0-9]+', '_', raw_class.strip().lower())

        st.subheader("Recommended Management Strategy 🌿")

        if "bacterial_spot" in major_class:
            st.write("""
**Chemical:** Copper Oxychloride 50% WP  
**Brands (Nepal):** Blitox, Blue Copper, Cu-50  
**Dosage:** 2–3 g per liter of water  
**Note:** Spray early morning or late evening to avoid leaf burn
            """)

        elif "early_blight" in major_class or "late_blight" in major_class:
            st.write("""
**Protective Chemical:** Mancozeb 75% WP  
**Brands:** Dithane M-45, Indofil M-45  
**Curative Chemical:** Metalaxyl 8% + Mancozeb 64% WP  
**Brands:** Krilaxyl, Ridomil Gold, Matco  
**Dosage:** 2 g per liter of water
            """)

        elif "leaf_mold" in major_class:
            st.write("""
**Chemical:** Carbendazim 50% WP  
**Brands:** Bavistin, Beve-50  
**Dosage:** 1–2 g per liter of water  
**Alternative:** Chlorothalonil (Kavach)
            """)

        elif "powdery_mildew" in major_class:
            st.write("""
**Chemical:** Wettable Sulphur 80% WP or Hexaconazole 5% EC  
**Brands:** Sulfex, Contaf, Sitara  
**Dosage:** 2 g per liter (Sulphur) or 2 ml per liter (Hexaconazole)
            """)

        elif "septoria" in major_class:
            st.write("""
**Chemical:** Chlorothalonil 75% WP  
**Brands:** Kavach, Ishan  
**Dosage:** 2 g per liter of water
            """)

        elif "spider_mites" in major_class or "two_spotted_spider_mite" in major_class:
            st.write("""
**Chemical:** Abamectin 1.8% or 1.9% EC  
**Brands:** Vertimec, Abacin, V-mectin  
**Dosage:** 0.5–1 ml per liter of water  
**Note:** Spray underside of leaves where mites hide
            """)

        elif "target_spot" in major_class:
            st.write("""
**Chemical:** Azoxystrobin 23% SC or Mancozeb  
**Brands:** Amistar, Mirador  
**Dosage:** 1 ml per liter of water
            """)

        elif "tomato_yellow_leaf_curl_virus" in major_class or "tylcv" in major_class:
            st.write("""
**Disease Type:** Viral (TYLCV) — no chemical cure  
**Vector Control:** Whitefly (Bemisia tabaci)  
**Chemical:** Imidacloprid 17.8% SL or Acetamiprid 20% SP  
**Brands:** Confidor, Media, Pride, Manik  
**Dosage:** 0.5 ml (Imidacloprid) or 0.5 g (Acetamiprid) per liter of water
            """)

        elif "tomato_mosaic_virus" in major_class:
            st.write("""
**Disease Type:** Tomato Mosaic Virus (ToMV) — viral disease, no chemical cure  
**Management Strategy:**  
- Remove and destroy infected plants to prevent spread  
- Practice crop rotation and avoid planting tomatoes in the same soil consecutively  
- Use resistant/tolerant varieties if available  
- Disinfect tools and equipment regularly  
- Control insect vectors (aphids, thrips) that may aid transmission  
**Note:** Focus on prevention and hygiene, as chemical sprays are ineffective against viruses
            """)
