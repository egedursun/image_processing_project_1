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

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

def get_model(input_shape):
    
    model = Sequential()
    
    model.add(
            Conv2D(
                    filters=32,
                    kernel_size=3,
                    activation='relu',
                    input_shape=input_shape,
                    )
            )
            
    model.add(
            BatchNormalization()
            )
    
    model.add(
            Conv2D(
                    filters=32,
                    kernel_size=3,
                    activation='relu',
                    )
             )
            
    model.add(
            BatchNormalization()
            )
    
    model.add(
            Conv2D(
                   filters=32,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   activation='relu',
                    )
            )
            
    model.add(
           BatchNormalization() 
            )
    
    model.add(
            Dropout(0.4)
            )
    
    model.add(
            Conv2D(
                 filters=64,
                 kernel_size=3,
                 activation='relu',
                    )
            )
            
    model.add(
           BatchNormalization() 
            )
    
    
    model.add(
            Conv2D(
                  filters=64,
                  kernel_size=3,
                  activation='relu',
                    )
            )
            
    model.add(
            BatchNormalization()
            )
    
    
    model.add(
            Conv2D(
                    filters=64,
                    kernel_size=5,
                    strides=2,
                    padding ='same',
                    activation='relu',
                    )
            )
            
    model.add(
            BatchNormalization()
            )
    
    model.add(
            Dropout(0.4)
            )
    
    model.add(
            Flatten()
            )
    
    model.add(
            Dense(
                   128,
                   activation='relu',
                    )
            )
            
    model.add(
            BatchNormalization()
            )
    
    model.add(
            Dropout(0.4)
            )
    
    model.add(
            Dense(
                    10,
                    activation='softmax',
                    )
            )
            
            
    model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=["accuracy"],
            )
    
    return model
            
    
    
