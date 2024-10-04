from ultralytics import YOLO
import cv2
from test_draw import draw_keypoints_and_skeleton


# Load a pretrained model
model = YOLO("yolo11n-pose.pt")

# Path to the video file
video_path = "fall_dataset/videos/video_1.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_counter = 0

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the frame
    results = model.predict(frame)

    # Draw the results on the frame
    for result in results:
        keypoints = result.keypoints.data[0]
        draw_keypoints_and_skeleton(frame, keypoints)


    # Display the frame
    cv2.imshow("Pose Estimation", frame)

    # Exit if 'q' is pressed
    # Pause every 5 frames for 1 second
    # to do: assign v11 model keypoint new numbering to main.py -> fall_detection
    frame_counter += 1
    if frame_counter % 2 == 0:
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Wait for 1 second (1000 ms)
            break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()