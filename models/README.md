# 📦 Models Directory

This directory is intended to store the machine learning or deep learning model files required by the Human Pose Estimation System. 

## Notes:
- MediaPipe downloads pose landmark models automatically when first run. However, if you're deploying this app in restricted or cloud environments (like Streamlit Cloud), you may need to pre-download the required `.tflite` models here to avoid permission issues.
- You can place models like `pose_landmark_lite.tflite` or `pose_landmark_heavy.tflite` in this folder and adjust the code to load from the local `models/` path.

## Structure:
models/
├── pose_landmark_lite.tflite # Example pose landmark model
└── README.md # This file


## Purpose:
Helps manage model files under version control and ensures consistent deployment across different environments.
