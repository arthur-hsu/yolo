import cv2
from ultralytics import YOLO
import numpy as np

def resize_and_pad_cv2(img, target_size=(300, 300), background_color=(0, 0, 0)):
    """
    Resize image and pad with specified background color to target size using OpenCV.
    """
    # Convert the background color from RGB to BGR (as OpenCV uses BGR)
    background_color = background_color[::-1]  # Reverse color order

    # Check if the current size is already the target size
    if img.shape[1] == target_size[0] and img.shape[0] == target_size[1]:
        return img

    # Calculate the ratio and the needed width and height
    ratio = min(target_size[0] / img.shape[1], target_size[1] / img.shape[0])
    new_size = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))

    # Resize the image using cv2
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

    # Create new image (canvas) with target size and fill with background color
    new_img = np.full((target_size[1], target_size[0], 3), background_color, dtype=np.uint8)

    # Calculate top-left corner position for pasting resized image
    paste_position = ((target_size[0] - new_size[0]) // 2,
                      (target_size[1] - new_size[1]) // 2)

    # Paste the resized image onto the canvas
    new_img[paste_position[1]:paste_position[1]+new_size[1],
            paste_position[0]:paste_position[0]+new_size[0]] = img

    print('after:',new_img.shape)
    return new_img


import os
os.environ["DISPLAY"]=':0'

# Load the YOLOv8 model
mod = './best.onnx'
model = YOLO(mod)
# model.to('mps')
# Open the video file
cap = cv2.VideoCapture(0)
# cap.set(3, 320)
# cap.set(4, 320)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    print('raw:',frame.shape)

    # frame = resize_and_pad_cv2(frame, target_size=(320, 320))
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # results = model(frame)
        results = model.predict(frame,augment=False, show=False, imgsz=(800,600), conf=0.25)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

