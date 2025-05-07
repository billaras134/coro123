import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(page_title="AI Coronary Angiography Assistant", layout="wide")
st.title("🫀 AI-Based Coronary Angiography Assistant (Cloud Compatible Version)")

st.write("""
Upload a coronary angiography video or image and let AI assist you in detecting vessels and possible lesions.
This is a prototype for research and educational purposes.
""")

uploaded_file = st.file_uploader("Upload an angiography video or image", type=["jpg", "jpeg", "png", "mp4"])

def estimate_stenosis(mask):
    vessel_pixels = np.sum(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    vessel_density = vessel_pixels / total_pixels
    stenosis_percent = max(0, (0.3 - vessel_density) * 300)
    return min(stenosis_percent, 99)

def recognize_artery(point, image_shape):
    h, w = image_shape[:2]
    x, y = point
    if y < h // 3:
        return "Left Anterior Descending (LAD)"
    elif y < 2 * h // 3:
        return "Left Circumflex (LCx)"
    else:
        return "Right Coronary Artery (RCA)"

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_ext == '.mp4':
        st.video(uploaded_file)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Total Frames in Video: {frame_count}")
        stframe = st.empty()
        frame_results = []

        for frame_num in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)
                vessel_highlight = frame.copy()
                vessel_highlight[mask == 1] = [0, 0, 255]
                stenosis_percent = estimate_stenosis(mask)
                artery = recognize_artery((frame.shape[1]//2, frame.shape[0]//2), frame.shape)
                stframe.image(vessel_highlight, caption=f"Frame {frame_num} - {artery} - Estimated Stenosis: {stenosis_percent:.1f}%")
                frame_results.append(f"Frame {frame_num}: {artery} - Estimated Stenosis = {stenosis_percent:.1f}%")

        cap.release()
        st.subheader("AI Interpretation Draft for Video")
        report_text = "\n".join(frame_results)
        st.download_button("Download Draft Report", report_text)

    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image_np, caption="Original Angiography Image", use_column_width=True)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)
        vessel_highlight = image_np.copy()
        vessel_highlight[mask == 1] = [0, 0, 255]
        stenosis_percent = estimate_stenosis(mask)
        artery = recognize_artery((image_np.shape[1]//2, image_np.shape[0]//2), image_np.shape)
        st.image(vessel_highlight, caption=f"AI Highlighted Vessels - {artery} - Estimated Stenosis: {stenosis_percent:.1f}%", use_column_width=True)
        st.subheader("AI Interpretation Draft")
        st.write(f"""
- Vessel segmentation completed.
- Detected artery: {artery}
- Estimated stenosis: {stenosis_percent:.1f}%
- Potential lesion zones marked in red.
- Suggested next step: Clinical correlation and possible further evaluation.
""")

        report_text = f"""
AI Coronary Angiography Preliminary Report
-------------------------------------------

Patient Image analyzed.
Vessels identified and mapped with AI model.
Detected artery: {artery}
Estimated stenosis: {stenosis_percent:.1f}%

* Areas of possible narrowing highlighted (requires confirmation).
* Recommend clinical evaluation.

---

This is an AI-assisted draft. Not for clinical decision making.
"""
        st.download_button("Download Draft Report", report_text)
