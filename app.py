import streamlit as st
import cv2
from ultralytics import YOLO
import time
import os
import smtplib
from email.message import EmailMessage

# ---------------- UI ----------------
st.set_page_config(page_title="AI Farm Guard", layout="wide")

if not os.path.exists("detections"):
    os.makedirs("detections")

if not os.path.exists("recordings"):
    os.makedirs("recordings")

# ---------------- EMAIL ----------------
def send_email(receiver_email, image_path, message):

    SENDER_EMAIL = "yourgmail@gmail.com"
    APP_PASSWORD = "your_app_password"

    msg = EmailMessage()
    msg["Subject"] = "🍅 AI Farm Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email

    msg.set_content(f"""
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Status: Detection Alert

{message}
""")

    with open(image_path, "rb") as f:
        img = f.read()
        msg.add_attachment(img, maintype="image", subtype="jpeg", filename="alert.jpg")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Email Error: {e}")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- UI INPUT ----------------
st.title("🍅 AI Tomato Farm Monitoring System")

camera_url = st.text_input("📷 IP Webcam URL")
email = st.text_input("📧 Email Address")

col1, col2 = st.columns(2)

with col1:
    conf = st.slider("🎯 Confidence", 0.0, 1.0, 0.5)

with col2:
    cooldown = st.slider("⏱ Alert Cooldown (sec)", 5, 60, 15)

record_video = st.checkbox("🎥 Record Video")

start = st.button("🚀 Start Monitoring")
stop = st.button("🛑 Stop")

video_placeholder = st.empty()
log_area = st.container()

# ---------------- MAIN ----------------
if start:

    if not camera_url or not email:
        st.error("❌ Please enter Camera URL and Email")
        st.stop()

    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        st.error("❌ Camera not reachable")
        st.stop()

    # VIDEO WRITER (recording)
    writer = None
    recording_path = None

    if record_video:
        recording_path = f"recordings/farm_{time.strftime('%Y%m%d-%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        frame_size = (640, 480)
        writer = cv2.VideoWriter(recording_path, fourcc, fps, frame_size)

    last_alert = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            st.error("Camera disconnected")
            break

        results = model(frame, conf=conf)
        annotated = frame.copy()

        counts = {"Red": 0, "Green": 0, "Turning": 0}
        detected = False

        for r in results:
            annotated = r.plot()

            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in counts:
                    counts[label] += 1
                    detected = True

        # SAVE VIDEO FRAME
        if writer is not None:
            writer.write(annotated)

        # DISPLAY
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # ---------------- ALERT ----------------
        if detected and time.time() - last_alert > cooldown:

            parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
            message = "🍅 Detection: " + ", ".join(parts)

            ts = time.strftime("%Y%m%d-%H%M%S")
            img_path = f"detections/{ts}.jpg"
            cv2.imwrite(img_path, frame)

            send_email(email, img_path, message)

            with log_area:
                st.image(img_path, caption=f"{message} | {ts}")

            last_alert = time.time()

        if stop:
            break

    cap.release()

    if writer is not None:
        writer.release()

        st.success("🎥 Recording saved!")

        with open(recording_path, "rb") as f:
            st.download_button(
                label="⬇ Download Recorded Video",
                data=f,
                file_name=os.path.basename(recording_path),
                mime="video/avi"
            )

    st.rerun()

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray;">
        Developed by <b>Sandesh Subedi</b>
    </div>
    """,
    unsafe_allow_html=True
)
