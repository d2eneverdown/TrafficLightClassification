import tensorflow as tf
from tensorflow.keras import layers, models
from ultralytics import YOLO

class SimpleCNN:
    def __init__(self, input_shape):
        self.model = self.create_model(input_shape)
    
    def create_model(self,input_shape):
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
    def compile_model(self,optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']):
        self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    def train(self,train_data,train_labels,epochs=10,batch_size=32,validation_data=None):
        self.model.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size,validation_data=validation_data)
    def evaluate(self,test_data,test_labels):
        return self.model.evaluate(test_data,test_labels)
    def predict(self,input_data):
        return self.model.predict(input_data)

