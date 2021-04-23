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
from model import get_model
from keras.models import load_model
import os
import tensorflow as tf
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold


#Determine the batch size, number of classes and total epochs
batch_size = 32
num_classes = 10
epochs = 12

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


#Define inputs and targets
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)


#Define K-Fold Cross validation (5 fold)
kfold = KFold(n_splits=5, shuffle=True)

lsss = []
accs = []

#Start training and record scores
fold_no = 1
for train, test in kfold.split(inputs, targets):
    
    model = get_model(input_shape)
    
    history = model.fit(inputs[train], targets[train],
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1
                        )
    
    scores = model.evaluate(inputs[test], targets[test], verbose = 1)
    print("_________________")
    print("Fold ", fold_no, " Loss is : ", scores[0])
    print("Fold ", fold_no, " Accuracy Score is : ", scores[1]*100, "%")
    print("_________________")
    
    lsss.append(scores[0])
    accs.append(scores[1])
    
    fold_no = fold_no + 1
    

#Print average loss and average accuracy for the k-fold validation
print("_________________")
print("AVERAGE LOSS : ", (sum(lsss)/len(lsss)))
print("AVERAGE ACCURACY : ", (sum(accs)/len(accs)))
print("_________________")




