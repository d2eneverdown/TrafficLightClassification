import cv2
import os
from util import predict_with_yolov8
from ultralytics import YOLO

model = YOLO("weights/trafficlight_v3.pt")
input_folder = "/Users/dubaoze/Workspace/Traffic_Light_Classification/test_data/test3"
# output_folder = "v2_output"
target_label = ['go','stop','warning']

for file in os.listdir(input_folder):
    img_path = os.path.join(input_folder,file)
    results = predict_with_yolov8(model,img_path)
    for idx, result in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = result.tolist()
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) /2
            center_y = (y1 + y2) /2
            class_name = model.names[int(cls)]
            
            if class_name in target_label:
                print(f"Image: {file}, Class: {class_name}, "
                      f"Coordinates: ({x1}, {y1}), ({x2}, {y2}), Confidence: {conf}")
  
