import cv2
import numpy as np
from math import sqrt, log 
import PosEstimationModule as pm
import time


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
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_filename}")

#main code

frames=[]
AvFrames=[]
detector = pm.poseDetector()
#cap = cv2.VideoCapture('videos/karate.mp4')
cap = cv2.VideoCapture(0)
start_time = time.time()
while True:
    success, img = cap.read()
    median = cv2.medianBlur(img, 5)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # median = imcv2.Canny(gray, 30, 100)
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    h, w, c = img.shape
    median = cv2.rectangle(median, (5, 5), (w, h), (247, 247, 247), -1)
    # print(h,w)
    if len(lmList) != 0:
        # print(lmList[14])
        bodyDrawing(lmList)

    fx=1
    fy=1
    img_ = cv2.resize(img, None, fx=0.5, fy=0.5)
    avathar_ = cv2.resize(median, None, fx=0.5, fy=0.5)
    cv2.imshow("Image", img_)
    cv2.imshow("Avathar", avathar_)
    AvFrames.append(avathar_)
    frames.append(img_)
    print("frame is added")
    cv2.waitKey(12)
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 10:
        break
save_video(frames, "outputTIKTOK.mp4", 30)
save_video(AvFrames, "outputAvathar.mp4", 30)




    