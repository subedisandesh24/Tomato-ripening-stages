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

# Supported Format Definitions
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
    h1, h2, h3 { font-weight: 700 !important; }
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
    st.error(f"Error loading models: {e}. Ensure 'fruit.pt', 'leafbest.pt', and 'leafdisease.pt' are in the repository.")
    st.stop()

# --- Helper Functions ---
COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "turning": (0, 255, 255), "default": (255, 255, 255)}

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
    vid_file = st.file_uploader("Upload Video", type=SUPPORTED_VIDEOS, key="t2_up")
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.info("Processing Video...")
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        progress_bar = st.progress(0)
        frame_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = det_model(frame, conf=0.35)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_n = det_model.names[int(box.cls[0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), get_color_bgr(cls_n), 3)
            out.write(frame)
            frame_cnt += 1
            if frame_cnt % 5 == 0: progress_bar.progress(min(frame_cnt / total_frames, 1.0))
        cap.release(); out.release()
        st.success("Complete!")
        with open(out_file, 'rb') as f:
            st.download_button("⬇️ Download Video", f.read(), file_name="annotated.mp4", mime="video/mp4")

# ==========================================
# TAB 3: DISEASE CLASSIFIER (DETECTOR + CLASSIFIER)
# ==========================================
with tab3:
    st.subheader("Leaf Disease Classification & Strategy")
    leaf_file = st.file_uploader("Upload Leaf", type=SUPPORTED_IMAGES, key="t3_up")
    if leaf_file:
        img = Image.open(leaf_file).convert("RGB")
        st.image(img, width=300)
        if st.button("Classify & Recommend", type="primary"):
            # Step 1: Detect leaf using leafbest.pt
            leaf_det_results = leaf_det_model(img, conf=0.30)
            if len(leaf_det_results[0].boxes) == 0:
                st.error("⚠️ **No leaf** detected. Please upload a clear image of a tomato leaf.")
            else:
                # Step 2: Classify with leafdisease.pt
                results = cls_model(img)
                names = results[0].names
                probs = results[0].probs
                top1_idx = probs.top1
                top1_name = names[top1_idx]
                top1_conf = probs.top1conf.item()

                st.divider()
                st.markdown(f"### 📊 Analysis Result: :red[{top1_name}] ({top1_conf:.2%})")
                
                # Step 3: RECOMMENDATION STRATEGY (Initial script logic)
                st.markdown(f"#### 🛡️ Recommended Strategy")
                major_class = top1_name.lower().replace(" ", "_")
                with st.container():
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                if "bacterial_spot" in major_class:
                    st.markdown("""
                    **Chemical:** Copper Oxychloride 50% WP  
                    **Brands (Nepal):** Blitox, Blue Copper, Cu-50  
                    **Dosage:** 2–3 g per liter of water  
                    **Note:** Spray early morning or late evening to avoid leaf burn
                    """)
                elif "early_blight" in major_class or "late_blight" in major_class:
                    st.markdown("""
                    **Protective Chemical:** Mancozeb 75% WP  
                    **Brands:** Dithane M-45, Indofil M-45  
                    **Curative Chemical:** Metalaxyl 8% + Mancozeb 64% WP  
                    **Brands:** Krilaxyl, Ridomil Gold, Matco  
                    **Dosage:** 2 g per liter of water
                    """)
                elif "leaf_mold" in major_class:
                    st.markdown("""
                    **Chemical:** Carbendazim 50% WP  
                    **Brands:** Bavistin, Beve-50  
                    **Dosage:** 1–2 g per liter of water  
                    **Alternative:** Chlorothalonil (Kavach)
                    """)
                elif "powdery_mildew" in major_class:
                    st.markdown("""
                    **Chemical:** Wettable Sulphur 80% WP or Hexaconazole 5% EC  
                    **Brands:** Sulfex, Contaf, Sitara  
                    **Dosage:** 2 g per liter (Sulphur) or 2 ml per liter (Hexaconazole)
                    """)
                elif "septoria" in major_class:
                    st.markdown("""
                    **Chemical:** Chlorothalonil 75% WP  
                    **Brands:** Kavach, Ishan  
                    **Dosage:** 2 g per liter of water
                    """)
                elif "spider_mites" in major_class or "two_spotted_spider_mite" in major_class:
                    st.markdown("""
                    **Chemical:** Abamectin 1.8% or 1.9% EC  
                    **Brands:** Vertimec, Abacin, V-mectin  
                    **Dosage:** 0.5–1 ml per liter of water  
                    **Note:** Spray underside of leaves where mites hide
                    """)
                elif "target_spot" in major_class:
                    st.markdown("""
                    **Chemical:** Azoxystrobin 23% SC or Mancozeb  
                    **Brands:** Amistar, Mirador  
                    **Dosage:** 1 ml per liter of water
                    """)
                elif "tomato_yellow_leaf_curl_virus" in major_class or "tylcv" in major_class:
                    st.markdown("""
                    **Disease Type:** Viral (TYLCV) — no chemical cure  
                    **Vector Control:** Whitefly (Bemisia tabaci)  
                    **Chemical:** Imidacloprid 17.8% SL or Acetamiprid 20% SP  
                    **Brands:** Confidor, Media, Pride, Manik  
                    **Dosage:** 0.5 ml (Imidacloprid) or 0.5 g (Acetamiprid) per liter of water
                    """)
                elif "tomato_mosaic_virus" in major_class or "mosaic_virus" in major_class:
                    st.markdown("""
                    **Disease Type:** Tomato Mosaic Virus (ToMV) — viral disease, no chemical cure  
                    **Management Strategy:**  
                    - Remove and destroy infected plants to prevent spread  
                    - Practice crop rotation and avoid planting tomatoes in the same soil consecutively  
                    - Use resistant/tolerant varieties if available  
                    - Disinfect tools and equipment regularly  
                    - Control insect vectors (aphids, thrips) that may aid transmission  
                    **Note:** Focus on prevention and hygiene, as chemical sprays are ineffective against viruses
                    """)
                else:
                    st.write("No specific chemical recommendation available for this class yet. Please consult a local horticulturist.")
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
            img_res = np.array(display_img)
            img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
            total_w = 0
            for box in res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w_cm = (x2-x1)/px_per_cm
                kg = ((4/3) * math.pi * ((w_cm/2)**3) / 1000000) * 900
                total_w += kg
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0,0,255), 2)
            st.image(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
            st.success(f"Total Weight: {total_w:.3f} kg")

st.markdown('<p class="footer">By Sandesh Subedi</p>', unsafe_allow_html=True)
