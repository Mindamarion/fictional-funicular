import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.set_page_config(page_title="Face Detection App", layout="centered")

# Title
st.title("👤 Face Detection using Viola–Jones (Haar Cascades)")

# Instructions
st.markdown("""
### Instructions
1. Upload an image **or capture from your webcam**.
2. Choose the rectangle color for detected faces.
3. Adjust **scaleFactor** and **minNeighbors** for detection sensitivity.
4. Click **Detect Faces** to run detection.
5. Save the processed image if you like the result.
""")

# Sidebar settings
st.sidebar.header("⚙️ Detection Settings")
scaleFactor = st.sidebar.slider("Scale Factor", 1.05, 2.0, 1.1, 0.05)
minNeighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)

# Color picker
color = st.sidebar.color_picker("Pick Rectangle Color", "#00FF00")
rect_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

# Input options: Upload OR Webcam
st.subheader("Choose Input")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Or capture from webcam")

# Get image
image = None
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
elif camera_input is not None:
    image = np.array(Image.open(camera_input).convert("RGB"))

if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if st.button("Detect Faces"):
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), rect_color, 2)

        # Show result
        st.image(image, caption="Detected Faces", channels="RGB")

        # Save + download
        result_filename = "detected_faces.jpg"
        cv2.imwrite(result_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        with open(result_filename, "rb") as file:
            st.download_button("💾 Download Processed Image", data=file, file_name=result_filename, mime="image/jpeg")
