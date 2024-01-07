from ultralytics import YOLO
from ultralytics import settings
import torch, os



# View all settings
print(settings)
home = os.getcwd()
# Update multiple settings
settings.update({'runs_dir': f'{home}/runs', 'tensorboard':True})
# Reset settings to default values
settings.reset()
# Return a specific setting
print(settings['runs_dir'])
model = YOLO('./best.pt')
device = "mps" if torch.backends.mps.is_available() else "cpu"  # 'mps' is for Apple Silicon GPU
model = model.to(device)

# Export the model to tflite format
success = model.export(imgsz=320,format='onnx',dynamic=True)
print(f"result of export: {success}")

