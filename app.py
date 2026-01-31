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

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 20px;
    }
    html, body, [class*="css"] {
        font-size: 18px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px;
        font-weight: bold;
    }
    .stButton button {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    .footer {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #eee;
        font-weight: bold;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 25px;
        border-radius: 10px;
        border-left: 8px solid #ff4b4b;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title ---
st.markdown('<p class="main-title">Advance Tomato Monitoring System</p>', unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    # 1. Fruit Detection
    fruit_model = YOLO("fruit.pt") 
    # 2. Leaf Object Detection (The Gatekeeper)
    leaf_det_model = YOLO("leafbest.pt")
    # 3. Leaf Disease Classification
    leaf_cls_model = YOLO("leafdisease.pt")
    
    return fruit_model, leaf_det_model, leaf_cls_model

try:
    det_model, leaf_det_model, cls_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'fruit.pt', 'leafbest.pt', and 'leafdisease.pt' are in the repository.")
    st.stop()

# --- Helper Functions ---
COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "turning": (0, 255, 255),
    "default": (255, 255, 255)
}

def get_color_bgr(cls_name):
    name_lower = cls_name.lower()
    if "red" in name_lower: return COLORS["red"]
    if "green" in name_lower: return COLORS["green"]
    if "turning" in name_lower: return COLORS["turning"]
    return COLORS["default"]

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🍎 1. Fruit Detector", 
    "🎥 2. Video Mode", 
    "🍃 3. Disease Classifier", 
    "⚖️ 4. Weight Estimation"
])

# ==========================================
# TAB 1: FRUIT DETECTOR
# ==========================================
with tab1:
    st.subheader("Fruit Detection")
    img_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png','bmp','webp'], key="t1_up")
    
    if img_file:
        original_pil = Image.open(img_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_pil, caption="Original Image")
        
        if col2.button("🔍 Detect Tomatoes", type="primary"):
            img_cv = np.array(original_pil)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            results = det_model(img_cv, conf=0.50, iou=0.5, agnostic_nms=True)
            counts = {"Red": 0, "Turning": 0, "Green": 0}
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = det_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                for key in counts.keys():
                    if key.lower() in cls_name.lower():
                        counts[key] += 1
                color = get_color_bgr(cls_name)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 4)
                cv2.putText(img_cv, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            final_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            with col2:
                st.image(final_img, caption="Detected Tomatoes")
                buf = io.BytesIO()
                Image.fromarray(final_img).save(buf, format="JPEG")
                st.download_button("⬇️ Download Result", data=buf.getvalue(), file_name="detected.jpg", mime="image/jpeg")
            st.subheader("Count Summary")
            counts["Total"] = sum(counts.values())
            st.dataframe(pd.DataFrame([counts]))

# ==========================================
# TAB 2: VIDEO MODE
# ==========================================
with tab2:
    st.subheader("Video Detection")
    vid_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], key="t2_up")
    
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Processing Video...")
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        frame_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = det_model(frame, conf=0.35, iou=0.5, agnostic_nms=True)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = det_model.names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), get_color_bgr(cls_name), 3)
            out.write(frame)
            frame_cnt += 1
            progress_bar.progress(frame_cnt / total_frames)
            
        cap.release()
        out.release()
        st.success("Processing Complete!")
        with open(out_file, 'rb') as f:
            st.download_button("⬇️ Download Video", f.read(), file_name="annotated_video.mp4", mime="video/mp4")

# ==========================================
# TAB 3: DISEASE CLASSIFIER (WITH GATEKEEPER)
# ==========================================
with tab3:
    st.subheader("Leaf Disease Classification & Strategy")
    leaf_file = st.file_uploader("Upload Leaf", type=['jpg','png','jpeg'], key="t3_up")
    
    if leaf_file:
        img = Image.open(leaf_file).convert("RGB")
        st.image(img, width=350, caption="Uploaded Image")
        
        if st.button("Classify & Recommend", type="primary"):
            # --- STEP 1: DETECT LEAF (The Gatekeeper) ---
            det_results = leaf_det_model(img, conf=0.30) 
            
            if len(det_results[0].boxes) == 0:
                # Logic: If leafbest.pt do not detect leaf then say "No leaf"
                st.error("❌ **No Leaf Detected**")
                st.warning("The system could not identify a tomato leaf in this image. Please ensure the leaf is clearly visible and try again.")
            else:
                # --- STEP 2: CLASSIFY DISEASE ---
                # A leaf was found, now proceed to classification using leafdisease.pt
                with st.spinner('Analyzing Leaf Health...'):
                    results = cls_model(img)
                    names = results[0].names
                    probs = results[0].probs
                    
                    top1_name = names[probs.top1]
                    top1_conf = probs.top1conf.item()
                    
                    st.divider()
                    st.success(f"**Result:** {top1_name} ({top1_conf:.2%})")
                    
                    st.markdown(f"#### 🛡️ Recommended Strategy for: :red[{top1_name}]")
                    major_class = top1_name.lower().replace(" ", "_")
                    
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                    if "bacterial_spot" in major_class:
                        st.markdown("**Chemical:** Copper Oxychloride 50% WP (Blitox) | **Dosage:** 2–3 g/L")
                    elif "early_blight" in major_class or "late_blight" in major_class:
                        st.markdown("**Chemical:** Mancozeb 75% WP (Dithane M-45) or Ridomil Gold | **Dosage:** 2 g/L")
                    elif "leaf_mold" in major_class:
                        st.markdown("**Chemical:** Carbendazim 50% WP (Bavistin) | **Dosage:** 1–2 g/L")
                    elif "powdery_mildew" in major_class:
                        st.markdown("**Chemical:** Wettable Sulphur (Sulfex) | **Dosage:** 2 g/L")
                    elif "spider_mites" in major_class:
                        st.markdown("**Chemical:** Abamectin (Vertimec) | **Dosage:** 0.5–1 ml/L")
                    elif "tomato_yellow_leaf_curl_virus" in major_class:
                        st.markdown("**Viral Disease:** Control Whiteflies using Imidacloprid (Confidor) 0.5 ml/L.")
                    elif "healthy" in major_class:
                        st.markdown("✅ **Healthy Leaf:** No chemical treatment needed. Maintain regular care.")
                    else:
                        st.write("Consult a local specialist for specific treatment of this condition.")
                    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: WEIGHT ESTIMATION
# ==========================================
with tab4:
    st.subheader("Weight Estimation")
    if 'w_image' not in st.session_state: st.session_state.w_image = None
    if 'points' not in st.session_state: st.session_state.points = []
    
    w_file = st.file_uploader("Upload Image", type=['jpg','png'], key="t4_up")
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
            if st.button("🗑️ Reset All"):
                st.session_state.points = []; st.rerun()
            real_len = st.number_input("Real Distance (cm) between points:", 5.0)
            zoom_width = st.slider("Width (px)", 300, 2000, 700, 50)
            calc_btn = st.button("⚖️ Calculate Weight", type="primary", disabled=(len(st.session_state.points) != 2))

        with col_img:
            base_w, base_h = st.session_state.w_image.size
            display_img = st.session_state.w_image.resize((zoom_width, int(zoom_width * (base_h/base_w))))
            draw = ImageDraw.Draw(display_img)
            for i, p in enumerate(st.session_state.points):
                draw.ellipse((p[0]-8, p[1]-8, p[0]+8, p[1]+8), fill=(255,0,0) if i==0 else (0,0,255))
            if len(st.session_state.points) == 2:
                draw.line(st.session_state.points, fill=(255, 255, 0), width=3)
                st.image(display_img)
            else:
                val = streamlit_image_coordinates(display_img, key="coords", width=zoom_width)
                if val:
                    pt = (val['x'], val['y'])
                    if not st.session_state.points or st.session_state.points[-1] != pt:
                        st.session_state.points.append(pt); st.rerun()

        if calc_btn and len(st.session_state.points) == 2:
            p1, p2 = st.session_state.points
            px_per_cm = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / real_len
            res = det_model(display_img, conf=0.35)
            img_res = np.array(display_img)
            img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
            total_weight = 0
            for i, box in enumerate(res[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w_cm = (x2-x1)/px_per_cm
                kg = ((4/3) * math.pi * ((w_cm/2)**3) / 1000000) * 900
                total_weight += kg
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(img_res, f"{kg:.3f}kg", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            st.image(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
            st.metric("Total Estimated Weight", f"{total_weight:.3f} kg")

st.markdown('<p class="footer">By Sandesh Subedi</p>', unsafe_allow_html=True)
