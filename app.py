import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import tempfile
import cv2
import os
import math
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Advanced Tomato Analysis")

# --- Custom Styles ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    # Ensure these files are in your GitHub repo
    det_model = YOLO("fruit.pt") 
    cls_model = YOLO("leafdisease.pt")
    return det_model, cls_model

try:
    det_model, cls_model = load_models()
except Exception as e:
    st.error(f"Error loading models. Ensure 'fruit.pt' and 'leafdisease.pt' are in the directory. Error: {e}")
    st.stop()

# --- Helpers ---
COLORS = {
    "Red": (255, 0, 0),       # Red
    "Green": (0, 255, 0),     # Green
    "Turning": (255, 255, 0)  # Yellow
}

def get_color(cls_name):
    # Default to white if class name doesn't match exactly
    for key in COLORS:
        if key.lower() in cls_name.lower():
            return COLORS[key]
    return (255, 255, 255)

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
    st.header("Fruit Detection & Counting")
    st.write("Upload an image to detect Red, Green, and Turning tomatoes (Confidence > 0.60).")
    
    img_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png','bmp','webp'], key="tab1_upload")
    
    if img_file:
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)
        
        # Run Detection
        results = det_model(img_array, conf=0.60)
        
        # Process Results
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        counts = {"Red": 0, "Turning": 0, "Green": 0}
        
        # Iterate through detections
        for box in results[0].boxes:
            coords = box.xyxy[0].tolist() # x1, y1, x2, y2
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = det_model.names[cls_id]
            
            # Map YOLO class names to your specific counters
            # Assuming your model returns 'Red', 'Green', 'Turning' strings
            matched_key = None
            for key in counts.keys():
                if key.lower() in cls_name.lower():
                    counts[key] += 1
                    matched_key = key
                    break
            
            # Draw Box
            color = COLORS.get(matched_key, (255, 255, 255))
            draw.rectangle(coords, outline=color, width=4)
            draw.text((coords[0], coords[1]-20), f"{cls_name} {conf:.2f}", fill=color, font=font)

        # 1. Output Table
        st.subheader("Detection Summary")
        total_fruits = sum(counts.values())
        counts['Total'] = total_fruits
        df_counts = pd.DataFrame([counts])
        st.table(df_counts)
        
        # 2. Annotated Image
        st.subheader("Annotated Image")
        st.image(image, caption="Detected Tomatoes", use_container_width=True)
        
        # 3. Download
        # Save to buffer
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        st.download_button("Download Annotated Image", data=buf.getvalue(), file_name="detected_tomatoes.jpg", mime="image/jpeg")

# ==========================================
# TAB 2: VIDEO MODE
# ==========================================
with tab2:
    st.header("Video Detection")
    st.warning("Note: Processing video on cloud servers can be slow. Short clips are recommended.")
    
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'], key="tab2_upload")
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        st.write("Processing video... please wait.")
        progress_bar = st.progress(0)
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output setup
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            results = det_model(frame, conf=0.60)
            
            # Annotate Frame using OpenCV
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = det_model.names[int(box.cls[0])]
                
                # Determine Color (BGR for OpenCV)
                color_rgb = get_color(cls_name)
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # Flip to BGR
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
                cv2.putText(frame, cls_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
            
            out.write(frame)
            frame_count += 1
            
            # Update progress every 10 frames to save resource
            if frame_count % 10 == 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        out.release()
        progress_bar.progress(1.0)
        
        # Display/Download
        st.success("Processing Complete!")
        
        # Read the processed file for download
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            
        st.download_button("Download Annotated Video", video_bytes, file_name="annotated_tomatoes.mp4", mime="video/mp4")

# ==========================================
# TAB 3: DISEASE CLASSIFIER
# ==========================================
with tab3:
    st.header("Leaf Disease Classification")
    leaf_file = st.file_uploader("Upload Leaf Image", type=['jpg','png','jpeg'], key="tab3_upload")
    
    if leaf_file:
        img = Image.open(leaf_file)
        st.image(img, caption="Uploaded Leaf", width=300)
        
        if st.button("Classify Disease"):
            results = cls_model(img)
            
            # Get Probabilities
            probs = results[0].probs
            top5_indices = probs.top5
            top5_conf = probs.top5conf.tolist()
            names = results[0].names
            
            top1_name = names[top5_indices[0]]
            top1_conf = top5_conf[0]
            
            st.divider()
            st.subheader(f"Prediction: :red[{top1_name}]")
            st.write(f"Confidence: **{top1_conf:.2%}**")
            
            st.write("### Top 3 Predictions")
            for i in range(min(3, len(top5_indices))):
                name = names[top5_indices[i]]
                conf = top5_conf[i]
                st.write(f"{i+1}. **{name}**: {conf:.2%}")
                st.progress(conf)
            
            st.info("💡 Recommendation Strategy: (This section will be updated with specific treatment plans for the detected disease).")

# ==========================================
# TAB 4: WEIGHT ESTIMATION
# ==========================================
with tab4:
    st.header("Fruit Weight Estimation")
    st.write("1. Upload Image. 2. Click two points to calibrate scale. 3. Enter real distance.")
    
    # State Management for Calibration
    if 'calib_points' not in st.session_state:
        st.session_state['calib_points'] = []
    
    weight_file = st.file_uploader("Upload Image for Weight", type=['jpg','png'], key="tab4_upload")
    
    if weight_file:
        image = Image.open(weight_file).convert("RGB")
        
        # --- Calibration Step ---
        st.write("### Step 1: Calibration")
        st.write("Click two points on the image below (e.g., a ruler or known object width).")
        
        # Custom component to get coordinates
        value = streamlit_image_coordinates(image, key="pil")
        
        if value:
            point = (value["x"], value["y"])
            # Add point if not duplicate of last click
            if not st.session_state['calib_points'] or st.session_state['calib_points'][-1] != point:
                st.session_state['calib_points'].append(point)
                
        # Show selected points
        points = st.session_state['calib_points'][-2:] # Get last 2 points
        
        if len(points) == 2:
            st.success(f"Points Selected: {points[0]} and {points[1]}")
            
            # Calculate Pixel Distance
            pixel_dist = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
            
            real_dist = st.number_input("Enter the real distance between these points (cm):", min_value=0.1, value=5.0)
            
            if st.button("Calculate Weights"):
                # Conversion Factor
                pixels_per_cm = pixel_dist / real_dist
                
                # Run Detection
                results = det_model(np.array(image), conf=0.5) # Lower threshold slightly for weight to catch more
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except:
                    font = ImageFont.load_default()

                fruit_data = []
                DENSITY = 900 # kg/m3
                
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_name = det_model.names[int(box.cls[0])]
                    
                    # Dimensions in Pixels
                    w_px = x2 - x1
                    h_px = y2 - y1
                    
                    # Dimensions in CM
                    # Approximating: Width of box is Diameter (W), Height is Length (L)
                    W_cm = w_px / pixels_per_cm
                    L_cm = h_px / pixels_per_cm
                    
                    # Volume Calculation (cm3)
                    # Check if spherical (difference < 15%)
                    if abs(W_cm - L_cm) < (0.15 * L_cm):
                        # Sphere
                        r = W_cm / 2
                        vol_cm3 = (4/3) * math.pi * (r**3)
                        shape = "Sphere"
                    else:
                        # Ellipsoid
                        vol_cm3 = (4/3) * math.pi * (L_cm/2) * ((W_cm/2)**2)
                        shape = "Ellipsoid"
                        
                    # Mass Calculation
                    # Convert cm3 to m3: / 1,000,000
                    vol_m3 = vol_cm3 / 1000000
                    mass_kg = vol_m3 * DENSITY
                    
                    fruit_data.append({
                        "ID": i+1,
                        "Stage": cls_name,
                        "Shape": shape,
                        "Length (cm)": round(L_cm, 2),
                        "Diameter (cm)": round(W_cm, 2),
                        "Weight (kg)": round(mass_kg, 4)
                    })
                    
                    # Annotate
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1), f"{mass_kg:.3f}kg", fill="white", font=font)
                
                # --- Output Results ---
                
                # 1. Annotated Image
                st.image(image, caption="Weight Estimation", use_container_width=True)
                
                # 2. Table
                if fruit_data:
                    df = pd.DataFrame(fruit_data)
                    
                    # Summary Metrics
                    total_yield = df["Weight (kg)"].sum()
                    count_summary = df["Stage"].value_counts().to_dict()
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Total Yield", f"{total_yield:.3f} kg")
                    c1.write(f"Total Fruits: {len(fruit_data)}")
                    c2.write("Count by Stage:")
                    c2.write(count_summary)
                    
                    st.dataframe(df)
                    
                    # Download Image
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    st.download_button("Download Weight Analysis Img", data=buf.getvalue(), file_name="tomato_weights.jpg", mime="image/jpeg")
                else:
                    st.warning("No tomatoes detected for weight estimation.")
                    
        else:
            st.info("Click two points on the image above to start calibration.")
