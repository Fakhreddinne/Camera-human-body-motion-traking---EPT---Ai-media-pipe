import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import tempfile

st.title("Capture Video on Streamlit Cloud")

frames = []  # To store captured frames

# Step 1: Capture Frames
capture_video = st.checkbox("Start Capturing Frames")
if capture_video:
    img_file_buffer = st.camera_input("Capture a frame")
    if img_file_buffer is not None:
        # Convert to a NumPy array
        bytes_data = img_file_buffer.getvalue()
        img_array = np.frombuffer(bytes_data, dtype=np.uint8)
        # Decode image to OpenCV format
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frames.append(frame)
        st.image(frame, channels="BGR")

# Step 2: Save as a Video
if st.button("Save and Upload Video"):
    if frames:
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_video:
            height, width, _ = frames[0].shape
            video_writer = cv2.VideoWriter(
                temp_video.name,
                cv2.VideoWriter_fourcc(*"XVID"),
                10,  # Frame rate
                (width, height),
            )

            # Write frames to the video
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

            st.success("Video successfully saved!")

            # Step 3: Provide a download button
            temp_video.seek(0)
            with open(temp_video.name, "rb") as file:
                st.download_button(
                    label="Download Video",
                    data=file,
                    file_name="output_video.avi",
                    mime="video/avi",
                )
    else:
        st.warning("No frames captured!")
