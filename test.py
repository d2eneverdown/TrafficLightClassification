import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
import util as ut
from models import create_cnn

warnings.filterwarnings('ignore')

target_classes = ['go', 'stop', 'warning']
color_map = {
    'go': 'green',
    'stop': 'red',
    'warning': 'yellow'
}
rgb_color_map = {
    'go': (0, 255, 0),
    'stop': (0, 0, 255),
    'warning': (0, 255, 255)
}


train_folder_list = [
    # 'daySequence1',
    # 'daySequence2',
    'dayTrain',
    # 'nightSequence1',
    # 'nightSequence2',
    # 'nightTrain',
    # 'sample-datClip6',
    # 'sample-nightClip1'
]
n_samples_per_class = 1000



train_annotation_df = get_annotarion_dataframe(train_folder_list)
target_classes = train_annotation_df['target'].unique() # 提取唯一目标类别
target_classes.sort()

index, counts = np.unique(train_annotation_df['target'],return_counts=True)
values = train_annotation_df['target'].value_counts()
colors = [color_map[target] for target in index]
# print(values.index)
print(index,counts)



train_annotation_df = resample_dataset(train_annotation_df,n_samples_per_class)
values = train_annotation_df['target'].value_counts()
colors = [color_map[target] for target in values.index]


img_values,labels = image_traffic_light_crop(train_annotation_df)





resized_imgs = []
size = (100,100)

for img in tqdm(img_values,desc="Resizing images"):
    padded = pad_to_square(img)
    resized = resize_image(padded,size)
    resized_imgs.append(resized)

resized_imgs[200].shape

X = resized_imgs
Y = labels
print(">>>>",len(X),len(Y))

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print("Training set size:{}",len(x_train))
print("Test set size:{}",len(x_test))

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
print(">>y",len(y_train_encoded),len(y_test_encoded))

x_train = np.array(x_train)
x_test = np.array(x_test)
print(">>x",len(x_train),len(x_test))

x_train = x_train.reshape(-1,100,100,3)
x_test = x_test.reshape(-1,100,100,3)
print(">>x",len(x_train),len(x_test))



input_shape = (100,100,3)
model = create_cnn(input_shape)
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

early_stopping = callbacks.EarlyStopping(monitor='val_loss',patience=3,verbose=True)
model.fit(x_train,y_train_encoded,epochs=100,validation_data=(x_test,y_test_encoded),callbacks=[early_stopping])

model.save('TLCv1.keras')

image_to_predict = np.expand_dims(x_test[0],axis=0)

predictions = model.predict(image_to_predict)
class_labels = ['Stop', 'Go', 'Warning']
class_index = np.argmax(predictions,axis=1)
predicted_class = [class_labels[i] for i in class_index]

predicted_class[0]