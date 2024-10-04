import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


SELECT = 1
weights = ['yolov7-w6-pose.pt', 'yolo11n-pose.pt']
WEIGHT = weights[SELECT]

def get_pose_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    weights = torch.load(WEIGHT, map_location=device)
    model = weights['model']
    _ = model.float().eval()
    if torch.cuda.is_available():
        model = model.half().to(device)
    return model, device

def get_pose(image, model, device):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)
    output = non_max_suppression_kpt(
        output, 0.25, 0.65, nc=model.yaml['nc'], 
        nkpt=model.yaml['nkpt'] if 'nkpt' in model.yaml else model.yaml['kpt_shape'][0],
        kpt_label=True
    )
    with torch.no_grad():
        output = output_to_keypoint(output)
    return image, output

def prepare_image(image):
    _image = image[0].permute(1, 2, 0) * 255
    _image = _image.cpu().numpy().astype(np.uint8)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    return _image

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return

    model, device = get_pose_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, output = get_pose(frame, model, device)
        _image = prepare_image(image)

        # for pose in output:
        #     plot_skeleton_kpts(_image, pose, 3)

        cv2.imshow('Real-time Pose Detection', _image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'fall_dataset/videos/video_1.mp4'  # Update this with the path to your video
    process_video(video_path)