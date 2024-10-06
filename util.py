import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.utils import resample
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from models import load_yolov8_model,predict_with_yolov8

# 加载Annotation数据
def get_annotarion_dataframe(train_data_folders):
    data_base_path = 'dataset/'
    annotation_list = list()
    for folder in [folder + '/' for folder in train_data_folders if os.listdir(data_base_path)]:
        annotation_path = ''
        if 'sample' not in folder:
            annotation_path = data_base_path + 'Annotations/Annotations/' + folder
            print("annotation_path:",annotation_path)
        else:
            annotation_path = data_base_path +folder*2
            print("annotation_path:",annotation_path)
        image_frame_path = data_base_path + folder*2

        df = pd.DataFrame()
        if 'Clip' in os.listdir(annotation_path)[0]:
            print("Clip folder detected")
            clip_list = os.listdir(annotation_path)
            print("clip_list:",clip_list)
            for clip_folder in clip_list:
                df = pd.read_csv(annotation_path + clip_folder + '/frameAnnotationsBOX.csv',sep=";")
                df['image_path'] = image_frame_path + clip_folder + '/frames/'
                annotation_list.append(df)
        else:
            df = pd.read_csv(annotation_path + 'frameAnnotationsBOX.csv',sep=";")
            df['image_path'] = image_frame_path + '/frames/'
            annotation_list.append(df)
    df = pd.concat(annotation_list)
    df = df.drop(['Origin file','Origin frame number','Origin track','Origin track frame number'],axis=1)
    df.columns = ['filename','target','x1','y1','x2','y2','image_path']
    df = df[df['target'].isin(target_classes)]
    df['filename'] = df['filename'].apply(lambda filename: re.findall("\/([\d\w-]*.jpg)",filename)[0])
    df = df.drop_duplicates().reset_index(drop=True) # 去重，重置索引 
    return df

def resample_dataset(annotation_df,n_samples):
    df_resample_list = list()
    for target in target_classes:
        df = annotation_df[annotation_df['target'] == target].copy()
        df_r = resample(df,n_samples=n_samples,random_state=42)
        df_resample_list.append(df_r)
    return pd.concat(df_resample_list).reset_index(drop=True)

def image_traffic_light_crop(df):
    print("Cutting out traffic signs...")
    img_values = []
    labels = []
    for index,row in tqdm(df.iterrows(),total=len(df)):
        image_path = row["image_path"]
        filename = row["filename"]
        target = row["target"]
        x1,x2,y1,y2 = row["x1"],row["x2"],row["y1"],row["y2"]
        img = cv2.imread(image_path + filename)
        cropped_img = img[y1:y2,x1:x2]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img_values.append(cropped_img)
        labels.append(target)
    return img_values,labels

def pad_to_square(img):
    height,width,_ = img.shape
    max_dim = max(height,width)
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) //2
    padded_img = np.pad(img,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant')
    return padded_img

def resize_image(img,target_size):
    resized_img = cv2.resize(img,target_size)
    return resized_img

def load_yolov8_model(model_path):
    model = YOLO(model_path)
    return model

def predict_with_yolov8(model,img_path):
    results = model.predict(source=img_path,save=True)
    return results

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