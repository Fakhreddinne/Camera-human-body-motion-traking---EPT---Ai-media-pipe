import cv2
import mediapipe as mp
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# Class to detect and analyze human poses using MediaPipe
class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        Initialize the pose detector.
        :param mode: Whether to treat the input images as static images or a video stream.
        :param smooth: Whether to apply smoothing to landmark detection.
        :param detectionCon: Minimum confidence value ([0.0, 1.0]) for pose detection to be considered successful.
        :param trackCon: Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be tracked successfully.
        """
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0  # Initialize previous time for FPS calculation

        # Initialize MediaPipe pose solutions
        self.mpDraw = mp.solutions.drawing_utils  # Utility to draw landmarks and connections
        self.mpPose = mp.solutions.pose  # Pose detection solution
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        """
        Detect pose landmarks in the given image.
        :param img: The input image in BGR format.
        :param draw: Whether to draw landmarks and connections on the image.
        :return: The image with optional drawings.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        self.results = self.pose.process(imgRGB)  # Process the image with MediaPipe pose

        # Draw landmarks and connections if detected and draw=True
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        """
        Get the positions of pose landmarks.
        :param img: The input image.
        :return: A list of landmark positions in the format [id, x, y].
        """
        self.lmList = []  # Initialize an empty list to store landmark positions
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape  # Get dimensions of the image
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                self.lmList.append([id, cx, cy])  # Append landmark ID and position
        return self.lmList

    def showFps(self, img):
        """
        Calculate and display frames per second (FPS) on the image.
        :param img: The input image.
        """
        cTime = time.time()  # Get the current time
        fbs = 1 / (cTime - self.pTime)  # Calculate FPS
        self.pTime = cTime  # Update previous time
        # Display FPS on the image
        cv2.putText(img, str(int(fbs)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculate the angle between three landmarks.
        :param img: The input image.
        :param p1: ID of the first landmark.
        :param p2: ID of the second (central) landmark.
        :param p3: ID of the third landmark.
        :param draw: Whether to draw the angle and connections on the image.
        :return: The calculated angle in degrees.
        """
        # Get the coordinates of the three landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        

          # Plot and save all pose landmarks on a 2D graduated plane over time
    def plotPose2D(self, frames):
        out = cv2.VideoWriter("graph.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        max_frames = 30 * 30  # Limit to 30 seconds of frames
        for i, frame in enumerate(frames):
            if i >= max_frames:
                break
            lmList = self.getPosition(frame)
            if lmList:
                plt.figure()
                x_vals = [lm[1] for lm in lmList]
                y_vals = [lm[2] for lm in lmList]
                plt.scatter(x_vals, y_vals, c='red')
                plt.plot(x_vals, y_vals, c='blue')
                plt.gca().invert_yaxis()
                plt.title('Pose Landmarks in 2D Plane')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.grid()
                plt.savefig("temp.png")
                plt.close()
                graph_frame = cv2.imread("temp.png")
                out.write(graph_frame)
        out.release()
# Function to save frames as a video
def save_video(frames, output_filename, fps):
    """
    Save a list of frames as a video file.
    :param frames: A list of frames.
    :param output_filename: Name of the output video file.
    :param fps: Frames per second for the output video.
    """
    if not frames:
        print("No frames to save.")
        return

    # Get dimensions of the frames
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)  # Write each frame to the video

    out.release()
    print(f"Video saved as {output_filename}")

# Main function to execute the pose detection pipeline
def main():
    frames = []  # List to store video frames
    detector = poseDetector()  # Initialize pose detector
    cap = cv2.VideoCapture(0)  # Open webcam feed

    start_time = time.time()  # Record start time
    while time.time() - start_time < 30:  # Capture frames for 30 seconds
        success, img = cap.read()  # Read frame from webcam
        if not success:
            break

        img = detector.findPose(img)  # Detect pose in the frame
        lmList = detector.getPosition(img)  # Get landmark positions
        detector.showFps(img)  # Display FPS on the frame

        cv2.imshow("Image", img)  # Show the frame with pose detection
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break

        frames.append(img)  # Add frame to the list

    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

    save_video(frames, "outputTIKTOK.mp4", 30)  # Save frames as video
    detector.plotPose2D(frames)  # Generate and save pose plot as video

if __name__ == "__main__":
    main()
