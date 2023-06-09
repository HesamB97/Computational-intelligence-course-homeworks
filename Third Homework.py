# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13SigvO4absTq2l7R4quERckujkDrX5U-
"""

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential 
import pandas as pd
x1 =pd.read_csv('weather2.csv')
y1 =pd.read_csv('flow2.csv')
import tensorflow as tf
y=np.array(y1.flow)
print(y1.shape)
print(x1.shape)

y1=np.expand_dims(y,1)
y_max=np.amax(y1,axis=0)
y_min=np.amin(y1,axis=0)
yN=(y1-y_min)/(y_max-y_min)
y_train=yN[:1626]
y_test=yN[-50:]
print(y_test.shape)

x_max=np.amax(x1,axis=0)
x_min=np.amin(x1,axis=0)
xN=(x1-x_min)/(x_max-x_min)
x_train=xN[:1626]
x_test=xN[-50:]
print(x_test.shape)
print(yN)

model=Sequential()
model.add(Dense(3, activation='relu',input_shape=(3, )))

model.add(Dense(1, activation='linear'))
model.compile(optimizer='Adam',  loss='mse')

history=model.fit(x_train,y_train,epochs=200);

import matplotlib.pyplot as plt
yy=(model.predict(x_train))

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
print(yy)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

er=(y_train-yy)/y_train
print((er))
erm=np.amin((er),axis=0)
print((erm))