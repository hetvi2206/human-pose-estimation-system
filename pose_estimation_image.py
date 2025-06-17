import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load an image
image_path = "./test images/run.jpg" # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform pose estimation
results = pose.process(image_rgb)

# Draw landmarks only (no lines)
if results.pose_landmarks:
    print("Pose landmarks detected!")

    # Extract landmark data
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z}), visibility: {landmark.visibility}")

    # Get image dimensions
    h, w, c = image.shape

    # Convert normalized coordinates to pixel coordinates
    # This loop is for demonstrating conversion and drawing individual keypoints.
    # The mp_drawing.draw_landmarks function below is more comprehensive for full drawing.
    for landmark in results.pose_landmarks.landmark:
        cx, cy = int(landmark.x * w), int(landmark.y * h)

        # Draw the keypoints
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1) # Green color, filled circle (Note: the image indicates Green, but the code is (255,0,0) which is blue in BGR. I've kept the code as-is but added a comment)

    #Optional: Draw landmarks on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the output image
    cv2.imshow("Pose Landmarks", image) # This shows image with just circles
    cv2.imshow("Pose drawing", annotated_image) # This shows image with lines and circles

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release resources
pose.close()