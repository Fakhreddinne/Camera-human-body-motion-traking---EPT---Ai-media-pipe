import streamlit as st
import cv2
import numpy as np
from math import log
from io import BytesIO
import PosEstimationModule as pm
import tempfile
import time

# Function to draw the body
def bodyDrawing(lmList, median):
    # Define body parts drawing logic
    distance = max(lmList[11][1] - lmList[12][1], 15)
    rect2center = (
        int((lmList[11][1] + lmList[12][1]) / 2),
        int((lmList[11][2] + lmList[12][2]) / 2),
    )
    cv2.line(median, (lmList[0][1], lmList[0][2]), rect2center, (0, 0, 0), 30)
    # Remaining body parts drawing logic...

# Save video to memory
def save_video(frames, fps):
    if not frames:
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = BytesIO()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        out = cv2.VideoWriter(tmpfile.name, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        output.write(open(tmpfile.name, "rb").read())
    return output

# Streamlit UI
st.title("Kinematic Control of the Human Body")
st.write("This app processes video frames to visualize pose estimation and outputs an avatar representation.")

option = st.sidebar.selectbox("Input Method", ["Upload a Video", "Use Webcam"])
frames = []
AvFrames = []

if option == "Upload a Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
else:
    st.write("Webcam is on.")
    while True:
        picture = st.camera_input('hey',disabled=not True, key=None)
        
        frames.append(picture)
        if len(frames) > 450:
            st.write("Webcam is off after 15 seconds.")
            break

    uploaded_file=save_video(frames, fps)


if st.sidebar.button("Start Processing"):
    if option == "Upload a Video" and not uploaded_file:
        st.error("Please upload a video file to proceed.")
    else:
        stframe1 = st.empty()
        stframe2 = st.empty()

        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_source = tfile.name
        else:
            video_source = 0  # Webcam

        cap = cv2.VideoCapture(video_source)
        detector = pm.poseDetector()

        while True:
            success, img = cap.read()
            if not success:
                break

            median = cv2.medianBlur(img, 5) # Un filtre médian est appliqué sur chaque frame pour réduire le bruit visuel.
            img = detector.findPose(img)
            lmList = detector.getPosition(img)
            h, w, _ = img.shape
            median = cv2.rectangle(median, (5, 5), (w, h), (247, 247, 247), -1)

            if lmList:
                bodyDrawing(lmList, median)

            img_ = cv2.resize(img, None, fx=0.3, fy=0.3)
            avathar_ = cv2.resize(median, None, fx=0.3, fy=0.3)

            stframe1.image(img_, channels="BGR", caption="Original Video Frame")
            stframe2.image(avathar_, channels="BGR", caption="Avatar Representation")

            frames.append(img_)
            AvFrames.append(avathar_)

        cap.release()
        st.success("Processing completed!")

        # Save and provide download links for videos
        original_video = save_video(frames, 30)
        avatar_video = save_video(AvFrames, 30)

        if original_video:
            st.download_button(
                "Download Original Processed Video",
                data=original_video.getvalue(),
                file_name="output_original.mp4",
                mime="video/mp4",
            )

        if avatar_video:
            st.download_button(
                "Download Avatar Processed Video",
                data=avatar_video.getvalue(),
                file_name="output_avatar.mp4",
                mime="video/mp4",
            )

st.write("---")
st.markdown("""
<div style="text-align: center;">
    <h6>Developed by Eng. Fakhreddine Annabi - Ecole Polytechnique de Tunisie</h6>
    <a href="https://www.linkedin.com/in/fakhreddine-annabi/" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" width="20" height="20" alt="LinkedIn">
    </a>
    <p>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
