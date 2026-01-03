import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pillow_heif
import cv2
import io
import tempfile
import numpy as np
import re

# Set page config
st.set_page_config(page_title="Tomato Monitoring System", layout="wide")
st.title("Tomato Monitoring System 🍅🧑‍🌾")

# -----------------------------
# Load models (Cached to prevent reloading on every interaction)
# -----------------------------
@st.cache_resource
def load_models():
    # Ensure you have fruit.pt and leafdisease.pt in your root folder
    f_model = YOLO("fruit.pt")
    d_model = YOLO("leafdisease.pt")
    return f_model, d_model

try:
    fruit_model, disease_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'fruit.pt' and 'leafdisease.pt' are in the directory.")
    st.stop()

# -----------------------------
# Define all 4 tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Fruit Image Detector",
    "📹 Fruit Video Detector",
    "🦠 Leaf Disease Classifier",
    "⚖️ Tomato Weight Estimator"
])

# -----------------------------
# Tab 1: Fruit Image Detector
# -----------------------------
with tab1:
    st.header("Fruit Image Detector")
    st.write("This tab will detect tomato ripening stages from images.")

    # Added unique key='tab1_uploader' to prevent duplicates
    uploaded = st.file_uploader("Upload a tomato image", type=["jpg", "png", "jpeg", "heic"], key="tab1_uploader")
    if uploaded:
        if uploaded.type == "image/heic":
            heif_file = pillow_heif.read_heif(uploaded.read())
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            img = Image.open(uploaded)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        results = fruit_model(np.array(img))
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detections", use_column_width=True)

        result_pil = Image.fromarray(result_img)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)
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

# -----------------------------
# Tab 2: Fruit Video Detector
# -----------------------------
with tab2:
    st.header("Fruit Video Detector")
    st.write("This tab will detect tomato ripening stages from video frames.")

    uploaded_video = st.file_uploader("Upload a tomato video", type=["mp4", "avi", "mov"], key="tab2_uploader")
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        input_path = tfile.name
        output_path = input_path.replace(".mp4", "_detected.mp4")

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Use 'avc1' or 'mp4v' for Streamlit compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25, (width, height))

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

        try:
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Detected Video",
                    data=f.read(),
                    file_name="tomato_detected.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Error preparing download: {e}")

# -----------------------------
# Tab 3: Leaf Disease Classifier
# -----------------------------
with tab3:
    st.header("Leaf Disease Classifier")
    st.write("This tab will classify tomato leaf diseases.")

    disease_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "png", "jpeg", "heic"], key="tab3_uploader")
    if disease_file:
        if disease_file.type == "image/heic":
            heif_file = pillow_heif.read_heif(disease_file.read())
            disease_img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            disease_img = Image.open(disease_file)

        st.image(disease_img, caption="Uploaded Leaf Image", use_column_width=True)

        # Run disease classification
        disease_results = disease_model(np.array(disease_img))
        probs = disease_results[0].probs

        # Top-3 predictions
        top3_indices = probs.top5[:3]
        st.subheader("Top-3 Disease Predictions 🌿")
        for idx in top3_indices:
            class_name = disease_model.names[idx]
            confidence = float(probs.data[idx])
            if idx == probs.top1:
                st.markdown(f"- 🔴 **{class_name}** → Confidence: `{confidence:.2f}`")
            else:
                st.write(f"- {class_name}: {confidence:.2f}")

        # Normalize top-1 class name
        raw_class = disease_model.names[probs.top1]
        major_class = re.sub(r'[^a-zA-Z0-9]+', '_', raw_class.strip().lower())

        st.subheader("Recommended Management Strategy 🌿")

        if "bacterial_spot" in major_class:
            st.write("**Chemical:** Copper Oxychloride 50% WP (Blitox, Blue Copper)")
        elif "early_blight" in major_class or "late_blight" in major_class:
            st.write("**Chemical:** Mancozeb 75% WP (Dithane M-45) or Metalaxyl+Mancozeb")
        elif "leaf_mold" in major_class:
            st.write("**Chemical:** Carbendazim 50% WP (Bavistin)")
        elif "powdery_mildew" in major_class:
            st.write("**Chemical:** Wettable Sulphur 80% WP or Hexaconazole")
        elif "septoria" in major_class:
            st.write("**Chemical:** Chlorothalonil 75% WP (Kavach)")
        elif "spider_mites" in major_class:
            st.write("**Chemical:** Abamectin 1.9% EC")
        elif "target_spot" in major_class:
            st.write("**Chemical:** Azoxystrobin 23% SC")
        elif "yellow_leaf_curl" in major_class or "tylcv" in major_class:
            st.write("**Virus:** Control Whitefly using Imidacloprid 17.8% SL")
        elif "mosaic_virus" in major_class:
            st.write("**Virus:** No cure. Remove infected plants. Control aphids.")
        else:
            st.write(f"No specific recommendation in database for: {raw_class}")

# -----------------------------
# Tab 4: Tomato Weight Estimator
# -----------------------------
with tab4:
    st.header("Tomato Weight Estimator")
    st.write("This tab will estimate tomato weight based on image and calibration.")

    # Added unique key='tab4_uploader'
    uploaded_weight_img = st.file_uploader("Upload a tomato image", type=["jpg", "png", "jpeg", "heic"], key="tab4_uploader")
    
    if uploaded_weight_img:
        if uploaded_weight_img.type == "image/heic":
            heif_file = pillow_heif.read_heif(uploaded_weight_img.read())
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            img = Image.open(uploaded_weight_img)

        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_np = np.array(img)

        st.subheader("Calibration: Pixel-to-CM")
        st.info("Measure a known object in the image or use the tomato width if known.")
        
        col1, col2 = st.columns(2)
        with col1:
            pixel_distance = st.number_input("Pixel distance (e.g., width of tomato in px)", min_value=1.0, value=100.0)
        with col2:
            real_distance_cm = st.number_input("Real distance (cm) for that pixel width", min_value=0.1, value=5.0)
            
        cm_per_pixel = real_distance_cm / pixel_distance if pixel_distance > 0 else 0

        if st.button("Calculate Weight"):
            results = fruit_model(img_np)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="Measured Tomatoes")

            stage_counts = {"Red": 0, "Turning": 0, "Green": 0}
            stage_weights = {"Red": 0.0, "Turning": 0.0, "Green": 0.0}

            total_weight = 0

            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = fruit_model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate dimensions in pixels
                W_px = x2 - x1
                L_px = y2 - y1

                # Convert to cm
                W = W_px * cm_per_pixel
                L = L_px * cm_per_pixel

                # Volume estimation (approximate as sphere or ellipsoid)
                if abs(W - L) < 0.1 * max(W, L):
                    r = W / 2
                    V_cm3 = (4/3) * np.pi * (r**3)
                else:
                    V_cm3 = (4/3) * np.pi * (L/2) * (W/2)**2

                # Density of tomato approx 0.9 g/cm3 (900 kg/m3)
                density = 0.95 # g/cm3
                M_g = V_cm3 * density
                
                total_weight += M_g

                if "red" in label.lower():
                    stage_counts["Red"] += 1
                    stage_weights["Red"] += M_g
                elif "turning" in label.lower():
                    stage_counts["Turning"] += 1
                    stage_weights["Turning"] += M_g
                elif "green" in label.lower():
                    stage_counts["Green"] += 1
                    stage_weights["Green"] += M_g

            st.write("### Estimated Weights")
            st.write(f"**Total Estimated Weight:** {total_weight:.2f} grams")
            st.json(stage_weights)
