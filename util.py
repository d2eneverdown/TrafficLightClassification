import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample

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