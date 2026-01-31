import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import tempfile
import cv2
import math
import io
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Advance Tomato Monitoring System")

# Supported Formats
SUPPORTED_IMAGES = ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff', 'tif', 'jfif', 'heic', 'heif']
SUPPORTED_VIDEOS = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'webm', 'flv', 'mpg', 'mpeg', '3gp']

# --- Custom CSS (Larger Fonts & Styling) ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 20px;
    }
    html, body, [class*="css"] { font-size: 18px; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px; font-weight: bold;
    }
    .stButton button { font-size: 20px !important; font-weight: 600 !important; }
    .footer {
        font-size: 16px; color: #666; text-align: center; margin-top: 50px;
        padding: 20px; border-top: 1px solid #eee; font-weight: bold;
    }
    .recommendation-box {
        background-color: #f0f2f6; padding: 25px; border-radius: 10px;
        border-left: 8px solid #ff4b4b; font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title ---
st.markdown('<p class="main-title">Advance Tomato Monitoring System</p>', unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        fruit_model = YOLO("fruit.pt") 
        leaf_det_model = YOLO("leafbest.pt")
        leaf_cls_model = YOLO("leafdisease.pt")
        return fruit_model, leaf_det_model, leaf_cls_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

det_model, leaf_det_model, cls_model = load_models()

# --- Helper Functions ---
COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "turning": (0, 255, 255), "default": (255, 255, 255)}
def get_color_bgr(cls_name):
    name_lower = cls_name.lower()
    if "red" in name_lower: return COLORS["red"]
    if "green" in name_lower: return COLORS["green"]
    if "turning" in name_lower: return COLORS["turning"]
    return COLORS["default"]

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🍎 1. Fruit Detector", "🎥 2. Video Mode", "🍃 3. Disease Classifier", "⚖️ 4. Weight Estimation"])

# ==========================================
# TAB 1: FRUIT DETECTOR
# ==========================================
with tab1:
    st.subheader("Fruit Detection")
    img_file = st.file_uploader("Upload Image", type=SUPPORTED_IMAGES, key="t1_up")
    if img_file:
        original_pil = Image.open(img_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1: st.image(original_pil, caption="Original Image")
        if col2.button("🔍 Detect Tomatoes", type="primary"):
            img_cv = np.array(original_pil)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            results = det_model(img_cv, conf=0.50, iou=0.5, agnostic_nms=True)
            counts = {"Red": 0, "Turning": 0, "Green": 0}
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = det_model.names[int(box.cls[0])]
                for key in counts.keys():
                    if key.lower() in cls_name.lower(): counts[key] += 1
                color = get_color_bgr(cls_name)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 4)
                cv2.putText(img_cv, f"{cls_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            final_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            with col2:
                st.image(final_img, caption="Detected Tomatoes")
            st.subheader("Count Summary")
            counts["Total"] = sum(counts.values())
            st.dataframe(pd.DataFrame([counts]))

# ==========================================
# TAB 2: VIDEO MODE
# ==========================================
with tab2:
    st.subheader("Video Detection")
    vid_file = st.file_uploader("Upload Video", type=SUPPORTED_VIDEOS, key="t2_up")
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = det_model(frame, conf=0.35)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            out.write(frame)
        cap.release(); out.release()
        st.success("Video Processed!")
        st.download_button("⬇️ Download Video", open(out_file, 'rb').read(), "processed.mp4")

# ==========================================
# TAB 3: DISEASE CLASSIFIER (FIXED GATEKEEPER)
# ==========================================
with tab3:
    st.subheader("Leaf Disease Classification")
    leaf_file = st.file_uploader("Upload Leaf", type=SUPPORTED_IMAGES, key="t3_up")
    if leaf_file:
        img = Image.open(leaf_file).convert("RGB")
        st.image(img, width=350)
        
        if st.button("Classify & Recommend", type="primary"):
            # 1. STEP: Detect leaf using leafbest.pt with 0.20 confidence
            # Results include only those boxes having confidence >= 0.20
            leaf_results = leaf_det_model(img, conf=0.20)
            
            # 2. STEP: Check if any leaf box exists
            if len(leaf_results[0].boxes) == 0:
                # If no leaf is detected OR confidence is below 0.20
                st.error("❌ **No leaf**")
                st.warning("Tomato leaf could not be detected with sufficient confidence (>0.20).")
            else:
                # 3. STEP: Leaf detected! Proceed to classification using leafdisease.pt
                with st.spinner('Leaf detected! Classifying disease...'):
                    results = cls_model(img)
                    names = results[0].names
                    probs = results[0].probs
                    top1_name = names[probs.top1]
                    top1_conf = probs.top1conf.item()

                    st.divider()
                    st.markdown(f"### 📊 Analysis: :red[{top1_name}] ({top1_conf:.2%})")
                    
                    # Recommendation Strategy (Original logic)
                    st.markdown("#### 🛡️ Recommended Strategy")
                    major_class = top1_name.lower().replace(" ", "_")
                    with st.container():
                        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                        if "bacterial_spot" in major_class:
                            st.markdown("**Chemical:** Copper Oxychloride 50% WP (Blitox, Blue Copper) | **Dosage:** 2–3 g/L")
                        elif "early_blight" in major_class or "late_blight" in major_class:
                            st.markdown("**Chemical:** Mancozeb 75% WP (Dithane M-45) or Ridomil Gold | **Dosage:** 2 g/L")
                        elif "leaf_mold" in major_class:
                            st.markdown("**Chemical:** Carbendazim 50% WP (Bavistin) | **Dosage:** 1–2 g/L")
                        elif "powdery_mildew" in major_class:
                            st.markdown("**Chemical:** Wettable Sulphur (Sulfex) | **Dosage:** 2 g/L")
                        elif "spider_mites" in major_class:
                            st.markdown("**Chemical:** Abamectin 1.8% EC (Vertimec) | **Dosage:** 0.5–1 ml/L")
                        elif "yellow_leaf_curl" in major_class or "tylcv" in major_class:
                            st.markdown("**Viral:** No cure. Control Whitefly with Imidacloprid (Confidor) 0.5 ml/L.")
                        elif "mosaic_virus" in major_class:
                            st.markdown("**Viral:** No cure. Remove infected plants immediately.")
                        elif "healthy" in major_class:
                            st.success("Leaf is healthy. No treatment needed.")
                        else:
                            st.write("Consult a specialist for this condition.")
                        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: WEIGHT ESTIMATION
# ==========================================
with tab4:
    st.subheader("Weight Estimation")
    if 'w_image' not in st.session_state: st.session_state.w_image = None
    if 'points' not in st.session_state: st.session_state.points = []
    w_file = st.file_uploader("Upload Image", type=SUPPORTED_IMAGES, key="t4_up")
    if w_file:
        if st.session_state.w_image is None or st.session_state.w_file_name != w_file.name:
            st.session_state.w_image = Image.open(w_file).convert("RGB")
            st.session_state.w_file_name = w_file.name
            st.session_state.points = []
    if st.session_state.w_image:
        col_ctrl, col_img = st.columns([1, 2])
        with col_ctrl:
            if st.button("↩️ Undo Point"):
                if st.session_state.points: st.session_state.points.pop(); st.rerun()
            if st.button("🗑️ Reset All"): st.session_state.points = []; st.rerun()
            real_len = st.number_input("Real Distance (cm):", 5.0)
            zoom_width = st.slider("Width (px)", 300, 2000, 700)
            calc_btn = st.button("⚖️ Calculate Weight", type="primary", disabled=(len(st.session_state.points) != 2))
        with col_img:
            base_w, base_h = st.session_state.w_image.size
            display_img = st.session_state.w_image.resize((zoom_width, int(zoom_width*(base_h/base_w))))
            draw = ImageDraw.Draw(display_img)
            for i, p in enumerate(st.session_state.points):
                draw.ellipse((p[0]-8, p[1]-8, p[0]+8, p[1]+8), fill=(255,0,0) if i==0 else (0,0,255))
            if len(st.session_state.points) < 2:
                val = streamlit_image_coordinates(display_img, key="coords", width=zoom_width)
                if val:
                    pt = (val['x'], val['y'])
                    if not st.session_state.points or st.session_state.points[-1] != pt:
                        st.session_state.points.append(pt); st.rerun()
            else: st.image(display_img)
        if calc_btn and len(st.session_state.points) == 2:
            p1, p2 = st.session_state.points
            px_per_cm = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / real_len
            res = det_model(display_img, conf=0.35)
            total_w = 0
            for box in res[0].boxes:
                w_cm = (map(int, box.xyxy[0])[2]-map(int, box.xyxy[0])[0])/px_per_cm
                total_w += ((4/3) * math.pi * ((w_cm/2)**3) / 1000000) * 900
            st.success(f"Total Weight: {total_w:.3f} kg")

st.markdown('<p class="footer">By Sandesh Subedi</p>', unsafe_allow_html=True)
