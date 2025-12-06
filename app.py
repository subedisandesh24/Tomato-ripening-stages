import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pillow_heif
import cv2

st.title("Tomato Ripening Stage Detector 🍅")

# Load your YOLOv8 model
model = YOLO("fruit best.pt")  # or "fruit best.pt" if you keep that filename

# Upload image (supports iPhone HEIC too)
uploaded = st.file_uploader("Upload a tomato image", type=["jpg", "png", "jpeg", "heic"])

if uploaded:
    # Handle HEIC images from iPhone
    if uploaded.type == "image/heic":
        heif_file = pillow_heif.read_heif(uploaded.read())
        img = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data
        )
    else:
        img = Image.open(uploaded)

    # Show original image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run YOLO detection
    results = model(img)
    result_img = results[0].plot()

    # Convert BGR → RGB to preserve true colors
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Show detection results
    st.image(result_img, caption="Detections", use_column_width=True)

    # Count tomatoes by class
    counts = {}
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        counts[label] = counts.get(label, 0) + 1

    st.subheader("Tomato Counts")
    st.write(counts)
