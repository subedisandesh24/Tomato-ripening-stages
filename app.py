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

# --- Custom CSS (Larger Fonts & Styling) ---
st.markdown("""
    <style>
    /* Main Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Global Font Increase */
    html, body, [class*="css"] {
        font-size: 18px;
    }
    
    /* Tab Labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px;
        font-weight: bold;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton button {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    /* Footer */
    .footer {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #eee;
        font-weight: bold;
    }
    
    /* Recommendation Box */
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

# --- Load Models (Now loading 3 models) ---
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
    "red": (0, 0, 255),       # Red (BGR)
    "green": (0, 255, 0),     # Green
    "turning": (0, 255, 255), # Yellow
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
            
            # Confidence 0.50
            results = det_model(img_cv, conf=0.50, iou=0.5, agnostic_nms=True)
            
            counts = {"Red": 0, "Turning": 0, "Green": 0}
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = det_model.names[cls_id]
                conf = float(box.conf[0])
                
                # Update Counts
                found_key = False
                for key in counts.keys():
                    if key.lower() in cls_name.lower():
                        counts[key] += 1
                        found_key = True
                
                # Draw Box & Text
                color = get_color_bgr(cls_name)
                label = f"{cls_name} {conf:.2f}"
                
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 4)
                
                font_scale = 1.5 
                thickness = 3
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img_cv, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)

            final_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(final_img, caption="Detected Tomatoes")
                
                final_pil = Image.fromarray(final_img)
                buf = io.BytesIO()
                final_pil.save(buf, format="JPEG")
                st.download_button("⬇️ Download Result", data=buf.getvalue(), file_name="detected.jpg", mime="image/jpeg")

            st.subheader("Count Summary")
            counts["Total"] = sum(counts.values())
            st.dataframe(pd.DataFrame([counts]))

# ==========================================
# TAB 2: VIDEO MODE
# ==========================================
with tab2:
    st.subheader("Video Detection")
    st.write("Upload a video. The system will process the **entire duration**.")
    vid_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], key="t2_up")
    
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Video Loaded: {total_frames} frames ({total_frames/fps:.1f} seconds). Processing...")
        
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_cnt = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = det_model(frame, conf=0.35, iou=0.5, agnostic_nms=True)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = det_model.names[int(box.cls[0])]
                color = get_color_bgr(cls_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, cls_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            out.write(frame)
            frame_cnt += 1
            
            if frame_cnt % 5 == 0 or frame_cnt == total_frames:
                percentage = frame_cnt / total_frames
                progress_bar.progress(min(percentage, 1.0))
                status_text.text(f"Processing: {int(percentage*100)}% complete ({frame_cnt}/{total_frames} frames)")
            
        cap.release()
        out.release()
        progress_bar.progress(1.0)
        status_text.success("Processing Complete!")
        
        with open(out_file, 'rb') as f:
            st.download_button("⬇️ Download Annotated Video", f.read(), file_name="annotated_video.mp4", mime="video/mp4")

# ==========================================
# TAB 3: DISEASE CLASSIFIER (WITH DETECTION GATEKEEPER)
# ==========================================
with tab3:
    st.subheader("Leaf Disease Classification & Strategy")
    leaf_file = st.file_uploader("Upload Leaf", type=['jpg','png'], key="t3_up")
    
    if leaf_file:
        img = Image.open(leaf_file)
        st.image(img, width=300)
        
        if st.button("Classify & Recommend"):
            # --- STEP 1: DETECT LEAF ---
            # Use leafbest.pt to check if there is a leaf
            # Using slightly lower conf (0.25) to be safe (we just want to know if a leaf exists)
            det_results = leaf_det_model(img, conf=0.25) 
            
            # Check if any boxes were found
            if len(det_results[0].boxes) == 0:
                st.error("⚠️ **No Tomato Leaf Detected**")
                st.warning("The system could not find a leaf in this image. Please upload a clear photo of a tomato leaf.")
                st.info("Note: This prevents analyzing random objects like houses or cars.")
                
            else:
                # --- STEP 2: CLASSIFY DISEASE ---
                # A leaf was found, so now we run the classifier
                
                results = cls_model(img)
                names = results[0].names
                probs = results[0].probs
                
                # Get Top Predictions
                top5_indices = probs.top5
                top5_conf = probs.top5conf.tolist()
                
                # Display Results
                st.divider()
                st.markdown("### 📊 Analysis Results")
                
                col_res1, col_res2 = st.columns([1, 1])
                with col_res1:
                    st.write("**Top 3 Confidence Levels:**")
                    for i in range(min(3, len(top5_indices))):
                        idx = top5_indices[i]
                        disease_name = names[idx]
                        confidence = top5_conf[i]
                        
                        if i == 0:
                            st.write(f"1. 🔴 **{disease_name}**: {confidence:.2%}")
                        else:
                            st.write(f"{i+1}. {disease_name}: {confidence:.2%}")
                
                top1_idx = probs.top1
                top1_name = names[top1_idx]
                
                st.markdown("---")
                st.markdown(f"#### 🛡️ Recommended Strategy for: :red[{top1_name}]")
                
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
    
    w_file = st.file_uploader("Upload Image", type=['jpg','png'], key="t4_up")
    
    if w_file:
        if st.session_state.w_image is None or st.session_state.w_file_name != w_file.name:
            st.session_state.w_image = Image.open(w_file).convert("RGB")
            st.session_state.w_file_name = w_file.name
            st.session_state.points = []

    if st.session_state.w_image:
        col_ctrl, col_img = st.columns([1, 2])
        
        with col_ctrl:
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("↩️ Undo Point"):
                    if st.session_state.points:
                        st.session_state.points.pop()
                        st.rerun()
            with c_btn2:
                if st.button("🗑️ Reset All"):
                    st.session_state.points = []
                    st.rerun()

            st.write(f"Points Selected: **{len(st.session_state.points)} / 2**")
            
            if len(st.session_state.points) < 2:
                st.info("👉 Step 1: Click the LEFT edge.\n👉 Step 2: Click the RIGHT edge.")
            else:
                st.success("✅ Reference Set!")

            real_len = st.number_input("Real Distance (cm) between points:", 5.0)
            
            st.divider()
            st.write("**Adjust Image Zoom:**")
            zoom_width = st.slider("Width (px)", min_value=300, max_value=2000, value=700, step=50)
            
            calc_btn = st.button("⚖️ Calculate Weight", type="primary", disabled=(len(st.session_state.points) != 2))

        with col_img:
            base_w, base_h = st.session_state.w_image.size
            ratio = base_h / base_w
            new_h = int(zoom_width * ratio)
            
            display_img = st.session_state.w_image.resize((zoom_width, new_h))
            draw = ImageDraw.Draw(display_img)
            
            for i, p in enumerate(st.session_state.points):
                color = (255, 0, 0) if i == 0 else (0, 0, 255)
                draw.ellipse((p[0]-8, p[1]-8, p[0]+8, p[1]+8), fill=color, outline="white", width=2)
            
            if len(st.session_state.points) == 2:
                draw.line(st.session_state.points, fill=(255, 255, 0), width=3)
            
            if len(st.session_state.points) < 2:
                val = streamlit_image_coordinates(display_img, key="coords", width=zoom_width)
                if val:
                    pt = (val['x'], val['y'])
                    if not st.session_state.points or st.session_state.points[-1] != pt:
                        st.session_state.points.append(pt)
                        st.rerun()
            else:
                st.image(display_img, caption="Reference Ready.")

        if calc_btn and len(st.session_state.points) == 2:
            st.divider()
            p1, p2 = st.session_state.points
            px_per_cm = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / real_len
            
            res = det_model(display_img, conf=0.35, iou=0.5, agnostic_nms=True)
            
            img_res = np.array(display_img)
            img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
            
            table_data = []
            total_weight = 0
            
            for i, box in enumerate(res[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = det_model.names[int(box.cls[0])]
                
                w_cm = (x2-x1)/px_per_cm
                l_cm = (y2-y1)/px_per_cm
                
                if abs(w_cm - l_cm) < (0.2 * l_cm):
                    vol = (4/3) * math.pi * ((w_cm/2)**3)
                else:
                    vol = (4/3) * math.pi * (l_cm/2) * ((w_cm/2)**2)
                
                kg = (vol/1000000)*900
                total_weight += kg
                
                table_data.append({
                    "No": i+1, 
                    "Stage": cls_name, 
                    "Weight (kg)": round(kg, 3)
                })
                
                cv2.rectangle(img_res, (x1, y1), (x2, y2), (0,0,255), 2)
                txt = f"{kg:.3f}kg"
                f_scale = 0.5 if zoom_width < 500 else 0.8
                cv2.putText(img_res, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0,0,255), 2)

            final_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
            
            st.image(final_res, caption="Weight Analysis (Zoomed View)")
            
            if table_data:
                table_data.append({
                    "No": "TOTAL",
                    "Stage": f"{len(table_data)} Fruits",
                    "Weight (kg)": f"{total_weight:.3f}"
                })
            
            st.markdown("### Detailed List")
            
            df = pd.DataFrame(table_data)
            
            def style_dataframe(x):
                df_styler = pd.DataFrame('', index=x.index, columns=x.columns)
                df_styler.iloc[:, 0] = 'font-weight: bold; background-color: #f0f2f6; color: black;'
                df_styler.iloc[-1, :] = 'font-weight: bold; background-color: #ffcccc; color: black;'
                df_styler.iloc[-1, 0] = 'font-weight: bold; background-color: #ff4b4b; color: white;'
                return df_styler

            st.dataframe(
                df.style.apply(style_dataframe, axis=None), 
                use_container_width=True, 
                hide_index=True
            )
            
            buf = io.BytesIO()
            Image.fromarray(final_res).save(buf, format="JPEG")
            st.download_button("⬇️ Download Image", buf.getvalue(), "weight_result.jpg", "image/jpeg")

# --- Footer ---
st.markdown('<p class="footer">By Sandesh Subedi</p>', unsafe_allow_html=True)
