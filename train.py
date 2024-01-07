from ultralytics import YOLO
from ultralytics import settings
import torch, os

# from roboflow import Roboflow
# rf = Roboflow(api_key='YOUR_API_KEY')
# project = rf.workspace('WORKSPACE').project('PROJECT')
# dataset = project.version(1).download('yolov8')



# View all settings
print(settings)
home = os.getcwd()
# Update multiple settings
settings.update({'runs_dir': f'{home}/runs', 'tensorboard':True})

# Reset settings to default values
settings.reset()
# Return a specific setting
print(settings['runs_dir'])
model = YOLO('yolov8n.pt')
device = "mps" if torch.backends.mps.is_available() else "cpu"  # 'mps' is for Apple Silicon GPU
model = model.to(device)


# results = model.train(data='https://app.roboflow.com/ds/qniqwxAE5G?key=8S2qLRgBvJ', epochs=1)
data = "https://app.roboflow.com/ds/gTqBlr66HA?key=L05ydp3vJZ"
results = model.train(data=data, epochs=60)
# Evaluate the model's performance on the validation set
results = model.val()


# Export the model to tflite format
success = model.export()
print(f"result of export: {success}")




# Perform object detection on an image using the model

import cv2
frame = cv2.imread("./test.jpg")
results = model(frame)

# Visualize the results on the frame
annotated_frame = results[0].plot()

# Display the annotated frame
cv2.imshow("YOLOv8 Inference", annotated_frame)
# Break the loop if 'q' is pressed
cv2.waitKey()







