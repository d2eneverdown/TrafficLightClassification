import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks

import util
from models import create_cnn

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

train_annotation_df = util.get_annotation_df(train_folder_list)
train_annotation_df_resample = util.resample_dataset(train_annotation_df,n_samples_per_class)
values = train_annotation_df['target'].value_counts()
colors = [color_map[target] for target in values.index]
img_values,labels = util.image_traffic_light_crop(train_annotation_df)

resized_imgs = []
size = (100,100)

for img in tqdm(img_values,desc="Resizing images"):
    padded = util.pad_to_square(img)
    resized = util.resize_image(padded,size)
    resized_imgs.append(resized)

X = resized_imgs
Y = labels

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.reshape(-1,100,100,3)
x_test = x_test.reshape(-1,100,100,3)

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

predictions = model.predict(x_test)
loss = model.evaluate(x_test,y_test_encoded,verbose=0)
class_labels = ['Stop', 'Go', 'Warning']
class_index = np.argmax(predictions,axis=1)
predicted_class = [class_labels[i] for i in class_index]

for i in range(len(predicted_class)):
    print(f"Image {i}: Predicted Class = {predicted_class[i]}")

print(f"Total Loss: {loss}")


# image_to_predict = np.expand_dims(x_test[0],axis=0)
# predictions = model.predict(image_to_predict)
# class_labels = ['Stop', 'Go', 'Warning']
# class_index = np.argmax(predictions,axis=1)
# predicted_class = [class_labels[i] for i in class_index]
# predicted_class[0]