import tensorflow as tf
from tensorflow.keras import layers, models
from ultralytics import YOLO

def create_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(16,(3,3),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32,(3,3),activation='relu'),

        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(3,activation='softmax')
    ])
    return model

def load_yolov8_model(model_path):
    model = YOLO(model_path)
    return model

def predict_with_yolov8(model,img_path):
    results = model.predict(source=img_path,save=True)
    return results