import streamlit as st
import cv2
import numpy as np
from math import sqrt, log 
import PosEstimationModule as pm
import time
from tempfile import NamedTemporaryFile

def bodyDrawing(lmList):
    # neck
    distance = lmList[11][1] - lmList[12][1]
    print(distance)
    rect2center = (int((lmList[11][1] + lmList[12][1]) / 2), int((lmList[11][2] + lmList[12][2]) / 2))
    cv2.line(median, (lmList[0][1], lmList[0][2]), rect2center, (0, 0, 0), 30)
    # upper body
    cv2.line(median, (lmList[11][1], lmList[11][2]), (lmList[12][1], lmList[12][2]), (0, 0, 0), 5)
    pts = np.array([[lmList[11][1], lmList[11][2]], [lmList[12][1], lmList[12][2]],
                    [lmList[24][1], lmList[24][2]], [lmList[23][1], lmList[23][2]]], np.int32)
    cv2.fillPoly(median, [pts], 255)
    # hand
    cv2.line(median, (lmList[11][1], lmList[11][2]), (lmList[13][1], lmList[13][2]), (0, 0, 0), 10)
    cv2.line(median, (lmList[13][1], lmList[13][2]), (lmList[15][1], lmList[15][2]), (0, 0, 0), 10)
    cv2.line(median, (lmList[12][1], lmList[12][2]), (lmList[14][1], lmList[14][2]), (0, 0, 0), 10)
    cv2.line(median, (lmList[14][1], lmList[14][2]), (lmList[16][1], lmList[16][2]), (0, 0, 0), 10)
    cv2.circle(median, (lmList[15][1], lmList[15][2]), 20, (0, 0, 255), cv2.FILLED)
    cv2.circle(median, (lmList[16][1], lmList[16][2]), 20, (0, 0, 255), cv2.FILLED)
    # head
    if distance < 15 :
        distance =15
    cv2.circle(median, (lmList[0][1], lmList[0][2]), int(5*log((distance**2))), (0, 0, 255), cv2.FILLED)

    # lower body
    cv2.line(median, (lmList[23][1], lmList[23][2]), (lmList[25][1], lmList[25][2]), (0, 0, 0), 10)
    cv2.line(median, (lmList[25][1], lmList[25][2]), (lmList[27][1], lmList[27][2]), (0, 0, 0), 10)
    cv2.line(median, (lmList[24][1], lmList[24][2]), (lmList[26][1], lmList[26][2]), (0, 0, 0), 10)
    cv2.line(median, (lmList[26][1], lmList[26][2]), (lmList[28][1], lmList[28][2]), (0, 0, 0), 10)
    cv2.circle(median, (lmList[27][1], lmList[27][2]), 20, (0, 0, 255), cv2.FILLED)
    cv2.circle(median, (lmList[28][1], lmList[28][2]), 20, (0, 0, 255), cv2.FILLED)

def save_video(frames, output_filename, fps):
    if not frames:
        st.error("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_filename}")

# Streamlit UI
st.title("Intelligent System for Kinematic Control of the Human Body")
st.write("This app processes video frames to visualize pose estimation and outputs an avatar representation.")

# Input Method
option = st.sidebar.selectbox("Choose input method:", ("Upload a Video", "Use Webcam"))

frames = []
AvFrames = []

if option == "Upload a Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_source = temp_file.name
else:
    video_source = 0  # Webcam


if st.sidebar.button("Start Processing"):
    if option == "Upload a Video" and not uploaded_file:
        st.error("Please upload a video file to proceed.")
    else:
        stframe1 = st.empty()
        stframe2 = st.empty()
        cap = cv2.VideoCapture(video_source)
        detector = pm.poseDetector()
        start_time = time.time()
        
        
        while True:
            print("2")
            success, img = cap.read()
            if not success:
                break

            # Call your original logic here
            median = cv2.medianBlur(img, 5)
            img = detector.findPose(img)
            lmList = detector.getPosition(img)
            h, w, c = img.shape
            median = cv2.rectangle(median, (5, 5), (w, h), (247, 247, 247), -1)
            if len(lmList) != 0:
                bodyDrawing(lmList)

            img_ = cv2.resize(img, None, fx=0.3, fy=0.3)
            avathar_ = cv2.resize(median, None, fx=0.3, fy=0.3)

            

            # Display live preview
            stframe1.image(img_, channels="BGR", caption="Original Video Frame")
            stframe2.image(avathar_, channels="BGR", caption="Avatar Representation")

            AvFrames.append(avathar_)
            frames.append(img_)
            

            # if time.time() - start_time > 10 or cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("3")
            #     break
            
        print("4")
        cap.release()
        cv2.destroyAllWindows()
        st.success("Processing completed!")

        # Save videos
        output_original = "outputTIKTOK.mp4"
        output_avatar = "outputAvathar.mp4"
        save_video(frames, output_original, 30)
        save_video(AvFrames, output_avatar, 30)

        # Download buttons
        st.download_button(
            label="Download Original Processed Video",
            data=open(output_original, "rb").read(),
            file_name=output_original,
            mime="video/mp4"
        )
        st.download_button(
            label="Download Avatar Processed Video",
            data=open(output_avatar, "rb").read(),
            file_name=output_avatar,
            mime="video/mp4"
        )

st.write("---")
st.markdown("""
    <div style="text-align: center; font-size: 10px; padding-bottom: 10px;">
        <h6>Developed by Eng. Fakhreddine Annabi - Ecole Polytechnique de Tunisie</h6>
        <p>
            <a href="https://www.linkedin.com/in/fakhreddine-annabi/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" width="20" height="20" alt="LinkedIn Profile">
            </a>
        </p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    <hr>
""", unsafe_allow_html=True)
