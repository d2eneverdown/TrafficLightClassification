import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import util
from models import load_yolov8_model,predict_with_yolov8

def predict_easy_cnn(img_folder_path):
    # 加载模型和标签
    class_labels = ['stop', 'go', 'warning']
    model = load_model("TLCv1.keras")

    # 存储所有图像的list
    img_arrays = []

    for img_file in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img_file)
        img = image.load_img(img_path, target_size=(100, 100))
        img_array = image.img_to_array(img)  # 转换为数组
        img_arrays.append(img_array)  # 添加到列表中

    img_arrays = np.array(img_arrays)
    predictions = model.predict(img_arrays)
    # 获取预测的类别索引
    class_index = np.argmax(predictions, axis=1)

    # 将索引映射到类别标签
    predicted_classes = [class_labels[i] for i in class_index]

    # 输出结果
    for img_file, pred_class in zip(os.listdir(img_folder_path), predicted_classes):
        print(f"Image: {img_file}, Predicted Class: {pred_class}")

def predict_yolov8(img_folder_path):
    # 加载模型和标签
    class_labels = ['stop', 'go', 'warning']
    model = load_yolov8_model('yolov8n.pt')
    for img_file in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path,img_file)
        results = predict_with_yolov8(model,img_path)
        for result in results[0].boxes.data:
            x1,y1,x2,y2,conf,cls = result.tolist()
            print(f"Image: {img_file}, Class: {model.names[int(cls)]},"
                  f"Coordinates: ({x1},{y1}),({x2},{y2}),Condidence:{conf}")

if __name__ == '__main__':
    # predict_easy_cnn("predict_data/test1")
    predict_yolov8("predict_data/test1")