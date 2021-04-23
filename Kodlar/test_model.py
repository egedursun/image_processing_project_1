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
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
import os
import tensorflow as tf
import seaborn as sns
import numpy as np


#height and width of the images
rows, cols = 28, 28
num_classes = 10


#Get the dataset
(_, _), (x_test, y_test) = mnist.load_data()


#Format the data for the channels
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
    
else:
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)
    
    
#Preprocess and normalize the data
x_test = x_test.astype('float32')

x_test = x_test / 255

y_test = keras.utils.to_categorical(y_test, num_classes)


#Load the model
cwd = os.getcwd()
path = os.path.join(cwd, 'model.hdf5')
model = load_model(path)


#Evaluate the model
score = model.evaluate(x_test, y_test, verbose=1)


#Show model performance
print("_________________")
print("Model Loss : ", score[0])
print("Model Accuracy : %", score[1]*100)
print("__________________")


#Create predictions 
predictions = model.predict(x_test)



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






