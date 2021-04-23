#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

EGE DOĞAN DURSUN
051700000006
EGE ÜNİVERSİTESİ
MÜHENDİSLİK FAKÜLTESİ
BİLGİSAYAR MÜHENDİSLİĞİ BÖLÜMÜ
GÖRÜNTÜ İŞLEME, 2020-2021 / GÜZ
PROJE 2


ULAŞILAN TEST ACCURACY DEĞERİ : "%99.62"
    
    
"""

#Import the necessary libraries
import keras
from keras.datasets import mnist
from keras import backend as K
from model import get_model
import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

#Determine the batch size, number of classes and total epochs
batch_size = 32
num_classes = 10
epochs = 30

#height and width of the images
rows, cols = 28, 28


#Get the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Format the data for the channels
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, rows, cols)
    x_test = x_test.reshape(x_test.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
    
else:
    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)
    

#Preprocess and normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Get the model we will use for the training
model = get_model(input_shape)

#Determine a file path to save the model
filepath = "model.hdf5"

#Create checkpointing callback object for saving the model
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')


#Start training the model
history = model.fit(
        x_train,
        y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose=1,
        validation_data = (x_test, y_test),
        callbacks = [checkpoint],
        )


#Evaluate the model
score = model.evaluate(x_test, y_test, verbose=1)


#Show model performance
print("_________________")
print("Model Loss : ", score[0])
print("Model Accuracy : %", score[1]*100)
print("__________________")


#Create predictions 
predictions = model.predict(x_test)


#Plot the training accuracy history
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy History')
plt.ylabel('Accuracy Value (%)')
plt.xlabel('Number of Epoch')
plt.show()


#Plot the validation accuracy history
plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy History')
plt.ylabel('Accuracy Value (%)')
plt.xlabel('Number of Epoch')
plt.show()


#Plot the training and validation accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy Value (%)')
plt.xlabel('Number of Epoch')
plt.show()


y_trues =[]
for i in range(0, len(y_test)):
    y_trues.append(np.argmax(y_test[i]))

y_preds = []
for j in range(0, len(predictions)):
    y_preds.append(np.argmax(predictions[j]))


#Create the confusion matrix
con_mat = tf.math.confusion_matrix(
        labels = y_trues,
        predictions = y_preds,
        ).numpy()


#Print the confusion matrix
print("\n HATA MATRİSİ : \n")
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat, annot=True,cmap=plt.cm.Blues, fmt='g')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
    
    
    
    
    







