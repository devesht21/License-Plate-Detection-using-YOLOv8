import os
from ultralytics import YOLO

ROOT_PATH = "C:\Users\deves\PycharmProjects\License Plate Detection using YOLOv8\data"

model = YOLO('yolov8s.yaml')

results = model.train(data=os.path.join(ROOT_PATH, 'data.yaml'), epochs=50)