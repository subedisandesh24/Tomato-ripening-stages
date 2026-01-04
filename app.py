import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import tempfile
import cv2
import math
import io
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Tomato Analysis AI")

# --- Styles ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    # Make sure fruit.pt and leafdisease.pt are in your GitHub folder
    det_model = YOLO("fruit.pt") 
    cls_model = YOLO("leafdisease.pt")
    return det_model, cls_model

try:
    det_model, cls_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Check if 'fruit.pt' exists.")
    st.stop()

# --- Helper Functions ---
COLORS = {
    "red": (255, 0, 0),       # Red
    "green": (0, 255, 0),     # Green
    "turning": (255, 255, 0), # Yellow
    "default": (255, 255, 255)
}

def get_color(cls_name):
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
    st.header("Fruit Detection")
    img_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png','bmp','webp'], key="t1_up")
    
    if img_file:
        # Open image
        original_image = Image.open(img_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        
        # Button to trigger detection
        if col2.button("🔍 Detect Tomatoes", type="primary"):
            # Lower confidence to 0.25 to ensure we see results
            results = det_model(original_image, conf=0.25)
            
            # Prepare drawing
            annotated_img = original_image.copy()
            draw = ImageDraw.Draw(annotated_img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            counts = {"Red": 0, "Turning": 0, "Green": 0, "Other": 0}
            
            # Process Detections
            boxes = results[0].boxes
            if len(boxes) == 0:
                st.warning("No tomatoes detected. Try a clearer image or one closer to the fruit.")
            else:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    cls_name = det_model.names[cls_id]
                    conf = float(box.conf[0])
                    
                    # Determine Color
                    color = get_color(cls_name)
                    
                    # Count
                    matched = False
                    for key in ["Red", "Green", "Turning"]:
                        if key.lower() in cls_name.lower():
                            counts[key] += 1
                            matched = True
                    if not matched:
                        counts["Other"] += 1

                    # Draw Box & Label
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    label = f"{cls_name} {conf:.2f}"
                    
                    # Draw text background for visibility
                    text_bbox = draw.textbbox((x1, y1), label, font=font)
                    draw.rectangle((x1, y1 - 25, text_bbox[2], y1), fill=color)
                    draw.text((x1, y1 - 25), label, fill="black", font=font)

                # Show Results
                with col2:
                    st.image(annotated_img, caption="Detected Tomatoes", use_container_width=True)
                    
                    # Download
                    buf = io.BytesIO()
                    annotated_img.save(buf, format="JPEG")
                    st.download_button("⬇️ Download Result", data=buf.getvalue(), file_name="detected.jpg", mime="image/jpeg")

                # Table
                st.subheader("Count Summary")
                counts["Total"] = sum(counts.values())
                st.dataframe(pd.DataFrame([counts]))

# ==========================================
# TAB 2: VIDEO MODE
# ==========================================
with tab2:
    st.header("Video Detection")
    vid_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], key="t2_up")
    
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.info("Processing video... (Showing first 300 frames to save time)")
        
        # Setup Output
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        frame_cnt = 0
        max_frames = 300 # Limit frames for cloud performance
        
        while cap.isOpened() and frame_cnt < max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            # Detect
            results = det_model(frame, conf=0.35)
            
            # Annotate
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = det_model.names[int(box.cls[0])]
                
                # Get Color (BGR for OpenCV)
                c = get_color(cls_name)
                color_bgr = (c[2], c[1], c[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
                cv2.putText(frame, cls_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                
            out.write(frame)
            frame_cnt += 1
            progress_bar.progress(frame_cnt / max_frames)
            
        cap.release()
        out.release()
        
        # Download
        with open(out_file, 'rb') as f:
            st.download_button("⬇️ Download Annotated Video", f.read(), file_name="tomato_video.mp4", mime="video/mp4")

# ==========================================
# TAB 3: DISEASE CLASSIFIER
# ==========================================
with tab3:
    st.header("Leaf Disease Classification")
    leaf_file = st.file_uploader("Upload Leaf", type=['jpg','png'], key="t3_up")
    
    if leaf_file:
        img = Image.open(leaf_file)
        st.image(img, width=250)
        
        if st.button("Analyze Leaf"):
            results = cls_model(img)
            names = results[0].names
            probs = results[0].probs
            
            top1 = names[probs.top1]
            conf = probs.top1conf.item()
            
            st.success(f"Diagnosis: **{top1}** ({conf:.1%})")
            
            with st.expander("Show Top 3 Confidence"):
                for i in range(min(3, len(probs.top5))):
                    idx = probs.top5[i]
                    st.write(f"{names[idx]}: {probs.top5conf[i]:.1%}")

# ==========================================
# TAB 4: WEIGHT ESTIMATION
# ==========================================
with tab4:
    st.header("Weight Estimation (Interactive)")
    st.info("1. Upload -> 2. Click 2 points on image -> 3. Enter real length -> 4. Calculate")
    
    # Initialize Session State for Points
    if 'w_image' not in st.session_state:
        st.session_state.w_image = None
    if 'points' not in st.session_state:
        st.session_state.points = []
        
    w_file = st.file_uploader("Upload Image", type=['jpg','png'], key="t4_up")
    
    # Handle File Upload
    if w_file:
        # Load image once and store in session
        if st.session_state.w_image is None or st.session_state.w_file_name != w_file.name:
            st.session_state.w_image = Image.open(w_file).convert("RGB")
            st.session_state.w_file_name = w_file.name
            st.session_state.points = [] # Reset points on new image

    if st.session_state.w_image:
        col_ctrl, col_img = st.columns([1, 2])
        
        with col_ctrl:
            st.write("### Controls")
            # Reset Button
            if st.button("🔄 Reset Points"):
                st.session_state.points = []
                st.rerun()
            
            st.write(f"Points Selected: **{len(st.session_state.points)} / 2**")
            
            real_len = st.number_input("Real Distance (cm)", value=5.0, min_value=0.1)
            
            calculate_btn = st.button("⚖️ Calculate Weight", disabled=(len(st.session_state.points) != 2))

        with col_img:
            # Prepare image for display (Draw existing points)
            display_img = st.session_state.w_image.copy()
            draw_display = ImageDraw.Draw(display_img)
            
            # Draw Points
            for p in st.session_state.points:
                r = 10 # Radius of click marker
                draw_display.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=(0, 120, 255), outline="white", width=2)
            
            # Draw Line if 2 points exist
            if len(st.session_state.points) == 2:
                p1 = st.session_state.points[0]
                p2 = st.session_state.points[1]
                draw_display.line([p1, p2], fill=(0, 120, 255), width=3)
            
            # INTERACTIVE COMPONENT
            # Only active if we need points. If we have 2, we just show the static image to stop clicking.
            if len(st.session_state.points) < 2:
                st.write("👇 **Click on the image to set reference points**")
                value = streamlit_image_coordinates(display_img, key="coords")
                
                # Detect Click
                if value:
                    new_point = (value['x'], value['y'])
                    # Prevent duplicate triggering on re-render
                    if not st.session_state.points or st.session_state.points[-1] != new_point:
                        st.session_state.points.append(new_point)
                        st.rerun()
            else:
                st.image(display_img, caption="Reference Set. Click Calculate.", use_container_width=True)

        # CALCULATION LOGIC
        if calculate_btn and len(st.session_state.points) == 2:
            st.divider()
            with st.spinner("Analyzing Size & Weight..."):
                p1, p2 = st.session_state.points
                pixel_dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                px_per_cm = pixel_dist / real_len
                
                # Run Model
                res = det_model(st.session_state.w_image, conf=0.25)
                
                final_img = st.session_state.w_image.copy()
                draw_final = ImageDraw.Draw(final_img)
                font_final = ImageFont.load_default()
                
                table_data = []
                total_weight = 0
                
                for i, box in enumerate(res[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_name = det_model.names[int(box.cls[0])]
                    
                    w_cm = (x2 - x1) / px_per_cm
                    l_cm = (y2 - y1) / px_per_cm
                    
                    # Logic: W is diameter, L is length
                    # If similar, Sphere. Else Ellipsoid.
                    if abs(w_cm - l_cm) < (0.2 * l_cm):
                        rad = w_cm / 2
                        vol = (4/3) * math.pi * (rad**3)
                        shape = "Sphere"
                    else:
                        vol = (4/3) * math.pi * (l_cm/2) * ((w_cm/2)**2)
                        shape = "Ellipsoid"
                    
                    weight_kg = (vol / 1000000) * 900 # density
                    total_weight += weight_kg
                    
                    # Add to Table
                    table_data.append({
                        "ID": i+1, "Stage": cls_name, 
                        "Dimensions (cm)": f"{l_cm:.1f}x{w_cm:.1f}", 
                        "Weight (kg)": round(weight_kg, 3)
                    })
                    
                    # Draw
                    draw_final.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw_final.text((x1, y1), f"{weight_kg:.3f}kg", fill="white")
                
                # Output
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(final_img, caption="Weight Analysis", use_container_width=True)
                    buf = io.BytesIO()
                    final_img.save(buf, format="JPEG")
                    st.download_button("Download Weight Image", buf.getvalue(), "weights.jpg", "image/jpeg")
                    
                with c2:
                    st.metric("Total Estimated Yield", f"{total_weight:.3f} kg")
                    st.dataframe(pd.DataFrame(table_data))
