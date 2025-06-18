import streamlit as st
import cv2
import tempfile
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils

# Local model path
LOCAL_MODEL_PATH = "models/pose_landmark_lite.tflite"

# Function for pose estimation on images
def pose_estimation_image(image_path):
    interpreter = load_interpreter()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (input_width, input_height))
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get landmark predictions
    landmarks = interpreter.get_tensor(output_details[0]['index'])

    h, w, _ = image_bgr.shape
    for lm in landmarks[0]:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        cv2.circle(image_bgr, (x, y), 5, (0, 255, 0), -1)

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), channels="RGB")



# Function for pose estimation on videos
def load_interpreter():
    interpreter = tflite.Interpreter(model_path="models/pose_landmark_lite.tflite")
    interpreter.allocate_tensors()
    return interpreter

def pose_estimation_video(video_path):
    interpreter = load_interpreter()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image_rgb, (input_width, input_height))
        input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get landmark predictions
        landmarks = interpreter.get_tensor(output_details[0]['index'])

        # Draw detected keypoints on original frame
        h, w, _ = frame.shape
        for lm in landmarks[0]:
            x = int(lm[0] * w)
            y = int(lm[1] * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# Streamlit UI
st.title("Human Pose Estimation System (TFLite)")

option = st.sidebar.selectbox("Choose an option", ["Video Pose Estimation"])

if option == "Video Pose Estimation":
    video_file_buffer = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    if video_file_buffer is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file_buffer.read())
        st.video(tfile.name)

        if st.button("Start Pose Estimation"):
            pose_estimation_video(tfile.name)
            

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

