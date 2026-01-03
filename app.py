import streamlit as st

st.set_page_config(page_title="Tomato Monitoring System", layout="wide")
st.title("Tomato Monitoring System 🍅🌿")

# Define all 4 tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Fruit Image Detector",
    "📹 Fruit Video Detector",
    "🦠 Leaf Disease Classifier",
    "⚖️ Tomato Weight Estimator"
])

# Tab 1
with tab1:
    st.header("Fruit Image Detector")
    st.write("This tab will detect tomato ripening stages from images.")

# Tab 2
with tab2:
    st.header("Fruit Video Detector")
    st.write("This tab will detect tomato ripening stages from video frames.")

# Tab 3
with tab3:
    st.header("Leaf Disease Classifier")
    st.write("This tab will classify tomato leaf diseases.")

# Tab 4
with tab4:
    st.header("Tomato Weight Estimator")
    st.write("This tab will estimate tomato weight based on image and calibration.")

