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

st.title("Tomato Monitoring System 🍅🌿")

# Load models
fruit_model = YOLO("fruit.pt")
disease_model = YOLO("leafdisease.pt")

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Fruit Image Detector",
    "📹 Fruit Video Detector",
    "🦠 Leaf Disease Classifier",
    "⚖️ Tomato Weight Estimator"
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

with tab4:
    uploaded_weight_img = st.file_uploader("Upload a tomato image", type=["jpg", "png", "jpeg", "heic"])
    if uploaded_weight_img:
        # Read image
        if uploaded_weight_img.type == "image/heic":
            heif_file = pillow_heif.read_heif(uploaded_weight_img.read())
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            img = Image.open(uploaded_weight_img)

        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_np = np.array(img)

        st.subheader("Calibration: Pixel-to-CM")
        st.write("Use either method below to set the conversion factor.")

        # Method A: Canvas drawing
        cm_per_pixel_A = 0.0
        if CANVAS_AVAILABLE:
            canvas = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#00AAFF",
                background_image=img,
                update_streamlit=True,
                height=img.height,
                width=img.width,
                drawing_mode="line",
                key="weight_calib_canvas"
            )
            real_distance_cm_A = st.number_input("Actual distance (cm) for drawn line", min_value=0.1, value=10.0)
            if canvas and canvas.json_data and canvas.json_data.get("objects"):
                last_obj = canvas.json_data["objects"][-1]
                if last_obj.get("type") == "line":
                    x1, y1 = last_obj["x1"], last_obj["y1"]
                    x2, y2 = last_obj["x2"], last_obj["y2"]
                    pixel_distance_A = float(np.hypot(x2 - x1, y2 - y1))
                    if pixel_distance_A > 0:
                        cm_per_pixel_A = real_distance_cm_A / pixel_distance_A
                        st.success(f"Canvas calibration: {cm_per_pixel_A:.4f} cm/pixel")

        # Method B: Manual input
        pixel_distance_B = st.number_input("Pixel distance between two points", min_value=1.0, value=100.0)
        real_distance_cm_B = st.number_input("Actual distance (cm)", min_value=0.1, value=10.0)
        cm_per_pixel_B = real_distance_cm_B / pixel_distance_B if pixel_distance_B > 0 else 0.0

        # Final conversion factor
        cm_per_pixel = cm_per_pixel_A if cm_per_pixel_A > 0 else cm_per_pixel_B
        if cm_per_pixel <= 0:
            st.warning("Please provide a valid calibration.")
            st.stop()

        # Run detection
        results = fruit_model(img_np)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        stage_counts = {"Red": 0, "Turning": 0, "Green": 0}
        stage_weights = {"Red": 0.0, "Turning": 0.0, "Green": 0.0}

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = fruit_model.names[cls].lower()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            W_px = x2 - x1
            L_px = y2 - y1
            W = W_px * cm_per_pixel
            L = L_px * cm_per_pixel

            if abs(W - L) < 0.1 * max(W, L):
                r = W / 2
                V_cm3 = (4/3) * np.pi * (r**3)
            else:
                V_cm3 = (4/3) * np.pi * (L/2) * (W/2)**2

            V_m3 = V_cm3 * 1e-6
            M_g = V_m3 * 900 * 1000

            if "red" in label:
                stage_counts["Red"] += 1
                stage_weights["Red"] += M_g
            elif "turning" in label:
                stage_counts["Turning"] += 1
                stage_weights["Turning"] += M_g
            elif "green" in label:
                stage_counts["Green"] += 1
                stage_weights["Green"] += M_g

            cv2.putText(annotated, f"{M_g:.1f} g", (int(x1), max(int(y1)-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(annotated, caption="Detections with Estimated Mass", use_column_width=True)

        result_pil = Image.fromarray(annotated)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Annotated Image", buf.getvalue(),
                           file_name="tomato_mass_estimation.png", mime="image/png")

        total_count = sum(stage_counts.values())
        total_weight = sum(stage_weights.values())

        st.sidebar.subheader("Detected Tomatoes")
        st.sidebar.metric("Red", stage_counts["Red"])
        st.sidebar.metric("Turning", stage_counts["Turning"])
        st.sidebar.metric("Green", stage_counts["Green"])
        st.sidebar.metric("All Stages", total_count)

        st.sidebar.subheader("Total Weights (g)")
        st.sidebar.metric("Red", f"{stage_weights['Red']:.1f}")
        st.sidebar.metric("Turning", f"{stage_weights['Turning']:.1f}")
        st.sidebar.metric("Green", f"{stage_weights['Green']:.1f}")
        st.sidebar.metric("All Stages", f"{total_weight:.1f}")

        st.subheader("Summary Table")
        st.table({
            "Stage": ["Red", "Turning", "Green", "All Stages"],
            "Count": [stage_counts["Red"], stage_counts["Turning"], stage_counts["Green"], total_count],
            "Total Weight (g)": [
                f"{stage_weights['Red']:.1f}",
                f"{stage_weights['Turning']:.1f}",
                f"{stage_weights['Green']:.1f}",
                f"{total_weight:.1f}"
            ]
        })
