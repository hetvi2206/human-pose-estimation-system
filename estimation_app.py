import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import time
import os

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils

# Local model path
LOCAL_MODEL_PATH = "models/pose_landmark_lite.tflite"

# Function for pose estimation on images
def pose_estimation_image(image):
    # Load local model using Pose with static image mode
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=0.5,
        model_asset_path=LOCAL_MODEL_PATH  # <-- Custom model path here
    ) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
            return annotated_image, results
        else:
            return image, None


# Function for pose estimation on videos
def pose_estimation_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        model_asset_path=LOCAL_MODEL_PATH
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

            frame_resized = cv2.resize(frame, (720, 480))
            stframe.image(frame_resized, channels="BGR")

        cap.release()

# Streamlit UI
st.title("📌 Human Pose Estimation App")
st.write("Upload an image or a video to detect human pose landmarks using MediaPipe.")

option = st.sidebar.selectbox("Select Mode", ("Image Pose Estimation", "Video Pose Estimation"))

if option == "Image Pose Estimation":
    image_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Original Image")

        annotated_image, results = pose_estimation_image(image)

        if results:
            st.success("Pose landmarks detected!")
            st.image(annotated_image, channels="BGR", caption="Pose Estimation Result")
        else:
            st.warning("No landmarks detected.")

elif option == "Video Pose Estimation":
    video_file = st.file_uploader("Upload a Video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        st.video(tfile.name)

        if st.button("Start Pose Estimation"):
            pose_estimation_video(tfile.name)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using MediaPipe, OpenCV and Streamlit")

