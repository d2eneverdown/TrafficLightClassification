import pandas as pd
import os

import os
import pandas as pd

def data_preprocessing():
    csv_file_base_path = '../dataset/Annotations/Annotations/'
    day_output_labels_base_dir = 'output_labels/day/'
    night_output_labels_base_dir = 'output_labels/night/'

    # 创建基础输出目录
    os.makedirs(day_output_labels_base_dir, exist_ok=True)
    os.makedirs(night_output_labels_base_dir, exist_ok=True)

    # 遍历 'day train' 和 'night train' 文件夹下的所有 CSV 文件
    train_folders = ['dayTrain', 'nightTrain']
    samples_per_folder = 200

    for folder in train_folders:
        folder_path = os.path.join(csv_file_base_path, folder)

        # 选择输出目录
        output_labels_base_dir = day_output_labels_base_dir if folder == 'dayTrain' else night_output_labels_base_dir

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):  # 确保这是一个文件夹
                # 为每个子文件夹创建输出路径
                output_labels_dir = os.path.join(output_labels_base_dir, subfolder)
                os.makedirs(output_labels_dir, exist_ok=True)

                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith('frameAnnotationsBOX.csv'):
                        csv_file_path = os.path.join(subfolder_path, file_name)
                        # 使用分号作为分隔符读取CSV文件，并设定第一行为表头
                        annotations = pd.read_csv(csv_file_path, sep=';', header=0)

                        count = 0  # 在每个文件开始时初始化计数器
                        
                        for _, row in annotations.iterrows():
                            if count >= samples_per_folder:
                                break  # 如果达到所需数量，则退出循环

                            filename = row['Filename']  # 使用列名访问文件名
                            class_name = row['Annotation tag']  # 使用列名访问标签
                            
                            # 获取坐标
                            x1 = int(row['Upper left corner X'])
                            y1 = int(row['Upper left corner Y'])
                            x2 = int(row['Lower right corner X'])
                            y2 = int(row['Lower right corner Y'])

                            # 将标签转换为类ID
                            class_id = 0 if class_name == 'stop' else 1 if class_name == 'go' else 2
                            
                            width = x2 - x1
                            height = y2 - y1
                            x_center = x1 + width / 2
                            y_center = y1 + height / 2

                            # 动态获取图像的宽度和高度
                            img_width, img_height = 1280, 960  # 替换为实际图像的宽度和高度，或者从图像文件获取

                            # 归一化坐标
                            x_center /= img_width
                            y_center /= img_height
                            width /= img_width
                            height /= img_height

                            # 直接在子文件夹中创建标签文件，而不是包含多余的路径
                            label_filename = os.path.join(output_labels_dir, os.path.splitext(os.path.basename(filename))[0] + '.txt')
                            with open(label_filename, 'a') as f:
                                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                            count += 1  # 增加计数器


def filter_train_data(input_folder_path):
    delete_count = 0
    for folder_name in os.listdir(input_folder_path):
        folder_path = os.path.join(input_folder_path,folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.jpg'):
                    label_file = os.path.splitext(file_name)[0] + '.txt'
                    label_path = os.path.join(folder_path,label_file)
                    if not os.path.isfile(label_path):
                        image_path = os.path.join(folder_path,file_name)
                        os.remove(image_path)
                        delete_count += 1
    print(f"Delete {delete_count} images without label file.")






if __name__ == '__main__':
    filter_train_data('train_data')
    # data_preprocessing()