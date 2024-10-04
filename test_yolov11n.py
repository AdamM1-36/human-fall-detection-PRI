from ultralytics import YOLO
import cv2
import math
from test_draw import draw_keypoints_and_skeleton


WEIGHT = 'yolo11n-pose.pt'

'''
0 - Nose
1 - Left Eye
2 - Right Eye
3 - Left Ear
4 - Right Ear
5 - Left Shoulder
6 - Right Shoulder
7 - Left Elbow
8 - Right Elbow
9 - Left Wrist
10 - Right Wrist
11 - Left Hip
12 - Right Hip
13 - Left Knee
14 - Right Knee
15 - Left Ankle
16 - Right Ankle
'''
def fall_detection(box, keypoint):
    box = box.detach().cpu().numpy()
    keypoint = keypoint.detach().cpu().numpy()
    xmin, ymin = box[0], box[1]
    xmax, ymax = box[2], box[3]
    left_shoulder_y = keypoint[5][1]
    left_shoulder_x = keypoint[6][0]
    right_shoulder_y = keypoint[6][1]
    right_shoulder_x = keypoint[6][0]
    left_body_y = keypoint[11][1]
    left_body_x = keypoint[11][0]
    right_body_y = keypoint[12][1]
    right_body_x = keypoint[12][0]
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
    left_foot_y = keypoint[15][1]
    right_foot_y = keypoint[16][1]
    dx = int(xmax) - int(xmin)
    dy = int(ymax) - int(ymin)
    difference = dy - dx

    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
            len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
            right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
            len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
            or difference < 0:
        return True, (xmin, ymin, xmax, ymax)
    return False, None

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
        boxes = result.boxes.data[0]
        keypoints = result.keypoints.data[0]
        draw_keypoints_and_skeleton(frame, keypoints)
        fall_detected, bbox = fall_detection(boxes, keypoints)

        if fall_detected:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                          thickness=5, lineType=cv2.LINE_AA)
            cv2.putText(frame, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Pose Estimation", frame)

    # Exit if 'q' is pressed
    # Pause every 4 frames to check quality
    frame_counter += 1
    if frame_counter % 4 == 0:
        if cv2.waitKey(200) & 0xFF == ord('q'):  # delay 200ms
            break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()