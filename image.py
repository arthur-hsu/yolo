from ultralytics import YOLO
from ultralytics import settings
import torch, os
import cv2

# View all settings
home = os.getcwd()
# Update multiple settings
settings.update({'runs_dir': f'{home}/runs', 'tensorboard':True})

# Reset settings to default values
settings.reset()
# Return a specific setting
print(settings['runs_dir'])
mod = './yolo-fastestv2-opt.onnx'
model = YOLO(mod)
device = "mps" if torch.backends.mps.is_available() else "cpu"  # 'mps' is for Apple Silicon GPU
# model = model.to(device)




frame = cv2.imread("./IMG_9910.JPG")
results = model(frame)

# import sys
# sys.exit(0)
# Visualize the results on the frame
annotated_frame = results[0].plot()

cv2.imshow("YOLOv8 Inference", annotated_frame)
cv2.waitKey()

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    plot = result.plot()  # Matplotlib plot of the detections
    print(f"{boxes =}\n\n")
    print(boxes.conf)
    targ=boxes.xyxy
    n_objects_detected = len(targ)
    print(f'Number of objects detected: {n_objects_detected}')
    print(targ)

    # print(f"{boxes =}\n\n")  # First bbox coordinates (xyxy format)
    # print(f"{probs =}\n\n")
    # print(f"{masks =}\n\n")
    # print(f"{keypoints =}\n\n")
