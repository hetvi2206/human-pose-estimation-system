import os
os.environ["MEDIAPIPE_CACHE_FILE_PATH"] = "/tmp"

import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import time

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def pose_estimation_image(image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            return annotated_image, results
        else:
            return image, None


def pose_estimation_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
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
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            frame_resized = cv2.resize(frame, (720, 480))
            stframe.image(frame_resized, channels="BGR")

        cap.release()


def pose_estimation_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    with mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            frame_resized = cv2.resize(frame, (720, 480))
            stframe.image(frame_resized, channels="BGR")

            # Exit the loop if 'Stop Webcam' checkbox is unchecked
            if not st.session_state.get("webcam_active", False):
                break

    cap.release()


# Streamlit UI
st.title("📌 Human Pose Estimation App")
st.write("Upload an image, video, or use your webcam to detect human pose landmarks using MediaPipe.")

option = st.sidebar.selectbox("Select Mode", ("Image Pose Estimation", "Video Pose Estimation", "Live Webcam Pose Estimation"))

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

elif option == "Live Webcam Pose Estimation":
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if start:
        st.session_state.webcam_active = True
        pose_estimation_webcam()

    if stop:
        st.session_state.webcam_active = False

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using MediaPipe, OpenCV and Streamlit")
