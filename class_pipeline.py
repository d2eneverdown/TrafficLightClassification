import numpy as np
import os
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import util
from models import load_yolov8_model,predict_with_yolov8

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def predict_yolov8(img_folder_path):
    # 加载模型和标签
    target_label = 'traffic light'
    traffic_light_images = []
    # model = load_yolov8_model('yolov8n.pt')
    model = load_yolov8_model('traffic_light_yolov8n.pt')
    
    # # 创建输出文件夹（如果不存在的话）
    # output_folder = 'output'
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    for img_file in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img_file)
        results = predict_with_yolov8(model, img_path)
        
        for idx, result in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = result.tolist()
            class_name = model.names[int(cls)]
            
            if class_name == target_label:
                print(f"Image: {img_file}, Class: {class_name}, "
                      f"Coordinates: ({x1}, {y1}), ({x2}, {y2}), Confidence: {conf}")
                
                # 裁剪图像
                image = Image.open(img_path)
                cropped_image = image.crop((x1, y1, x2, y2))
                
                traffic_light_images.append(cropped_image)

                # # 使用序号为裁剪图像命名
                # cropped_image_path = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_cropped_{idx}.png")
                # # 保存裁剪后的图像
                # cropped_image.save(cropped_image_path)
                # # 将裁剪图像的路径添加到列表
                # traffic_light_images.append(cropped_image_path)

    return traffic_light_images  # 返回包含裁剪后图像地址的列表

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

def img_SR(img_list):
    model = RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4)
    model_path = 'weights/RealESRGAN_x4plus.pth'
    SR_img = []
    upsamples = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        half=False,
    )
    os.makedirs('/output/realesrgan_output',exist_ok=True)

    for idx,cropped_image in enumerate(img_list):
        img_array = np.array(cropped_image)
        output,_ = upsamples.enhance(img_array,outscale=4)
        padded_img = pad_img2square(output)
        save_path = os.path.join('realesrgan_output', f'SRout_{idx}.png')
        cv2.imwrite(save_path, padded_img)
        print(f'Saved to {save_path}')

        SR_img.append(save_path)

    # paths = img_list
    # for idx,path in enumerate(paths):
    #     imgname,extension = os.path.splitext(os.path.basename(path))
    #     print(f"Processing SR {imgname}...")
    #     img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    #     output,_ = upsamples.enhance(img,outscale=4)
    #     save_path = os.path.join('realesrgan_output',f'{imgname}_out.{extension}')
    #     cv2.imwrite(save_path,output)
    #     print(f'Saved to {save_path}')
    #     SR_img.append(save_path)
    return SR_img

def pad_img2square(img):
    h, w, _ = img.shape  # 获取图像的高度（h）和宽度（w）

    if h == w:  # 如果图像已经是正方形
        return img  # 直接返回图像

    elif h > w:  # 如果高度大于宽度
        pad_img = np.zeros((h, h, 3), dtype=np.uint8)  # 创建一个新图像，尺寸为 (h, h)
        pad_img[:h, (h - w) // 2:(h - w) // 2 + w, :] = img  # 将原图像放入新图的中央
        return pad_img  # 返回填充后的图像

    else:  # 如果宽度大于高度
        pad_img = np.zeros((w, w, 3), dtype=np.uint8)  # 创建一个新图像，尺寸为 (w, w)
        pad_img[(w - h) // 2:(w - h) // 2 + h, :w, :] = img  # 将原图像放入新图的中央
        return pad_img  # 返回填充后的图像


def main():
    # 使用yolov8模型对图片进行预检测
    traffic_img =predict_yolov8('predict_data/test1')
    print((traffic_img))
    SR_img =  img_SR(traffic_img)
    # 使用模型进行预测
    predict_easy_cnn('pad_output')


if __name__ == '__main__':
    main()