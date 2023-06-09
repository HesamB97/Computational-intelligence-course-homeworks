# -*- coding: utf-8 -*-
"""Hesam_Boroomand.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MW63Pk3FJywUS8Us6nb3pbm1ItUNUmZi

# تمرین اول_شبکه_اول
"""

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd
import tensorflow as tf

ds =pd.read_csv('train1.csv')
y=np.array(ds.price_range)
y=y[:1800]
y=np.expand_dims(y,1)
y= tf.keras.utils.to_categorical(y, 4)
y_test=np.array(ds.price_range)
y_test=y_test[-200:]
y_test=np.expand_dims(y_test,1)

ds_max=np.amax(ds,axis=0)
ds_min=np.amin(ds,axis=0)
ds=(ds-ds_min)/(ds_max-ds_min)
x=np.array(ds.drop(['price_range'],axis=1))
x=x[:1800]
y_test= tf.keras.utils.to_categorical(y_test, 4)
print(y_test)

x_test=np.array(ds.drop(['price_range'],axis=1))
x_test=x_test[-200:]

model=Sequential()
model.add(Dense(50, activation='sigmoid',input_shape=(20, )))
model.add(Dense(50, activation='sigmoid',input_shape=(50, )))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='mse')

history=model.fit(x,y,epochs=400);

import matplotlib.pyplot as plt
yy=model.predict(x_test)
yy2= tf.keras.utils.to_categorical(yy, 4)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

yy=np.around(yy)
print(yy)

print(y_test)

"""# تمرین اول_شبکه_دوم """

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from sklearn.cluster import KMeans
ds =pd.read_csv('train1.csv')

import pandas as pd
import tensorflow as tf

ds =pd.read_csv('train1.csv')
y=np.array(ds.price_range)
y=y[:1800]
y=np.expand_dims(y,1)
y_test=np.array(ds.price_range)
y_test=y_test[-200:]
y_test=np.expand_dims(y_test,1)

ds_max=np.amax(ds,axis=0)
ds_min=np.amin(ds,axis=0)
ds=(ds-ds_min)/(ds_max-ds_min)
x=np.array(ds.drop(['price_range'],axis=1))
x=x[:1800]
print(y_test)

x_test=np.array(ds.drop(['price_range'],axis=1))
x_test=x_test[-200:]

class RBN:
  def __init__(self,x,y,neurons=1500, b=1):
    self.x=x
    self.y=y
    self.neurons=neurons
    self.b=b
    km=KMeans(neurons)
    km.fit(x)
    self.centers = km.cluster_centers_
    self.xx=np.array([[self.gaussian(x[i],self.centers[j],b)for j in range(len(self.centers))] for i in range(len(x)) ])
    self.xx=np.concatenate((np.ones((len(self.xx),1)), self.xx), axis=1)
    self.w2= 0


  def fit(self):
    self.w2=np.linalg.inv(self.xx.T.dot(self.xx)).dot(self.xx.T).dot(self.y)

  def gaussian(self,x,center, b):
      n=np.linalg.norm(x-center)*b
      return np.exp(-n*n)

  def predict(self,x):
    xx=np.array([[self.gaussian(x[i],self.centers[j],self.b)for j in range(len(self.centers))] for i in range(len(x)) ])
    xx=np.concatenate((np.ones((len(xx),1)), xx), axis=1)
    return xx.dot(self.w2)

rbn= RBN(x,y,1500)
rbn.fit()
o=rbn.predict(x_test)

print(np.around(o)-y_test)

"""# تمرین اول_شبکه_سوم """

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

import pandas as pd
import tensorflow as tf

ds =pd.read_csv('train1.csv')
y=np.array(ds.price_range)
y=y[:1800]
y=np.expand_dims(y,1)
y= tf.keras.utils.to_categorical(y, 4)
y_test=np.array(ds.price_range)
y_test=y_test[-200:]
y_test=np.expand_dims(y_test,1)

ds_max=np.amax(ds,axis=0)
ds_min=np.amin(ds,axis=0)
ds=(ds-ds_min)/(ds_max-ds_min)
x=np.array(ds.drop(['price_range'],axis=1))
x=x[:1800]
y_test= tf.keras.utils.to_categorical(y_test, 4)
print(y_test)

x_test=np.array(ds.drop(['price_range'],axis=1))
x_test=x_test[-200:]

model=Sequential()
model.add(Dense(10, activation='sigmoid',input_shape=(20, )))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='mse')

history=model.fit(x,y,epochs=400);

import matplotlib.pyplot as plt
yy=model.predict(x_test)
yy2= tf.keras.utils.to_categorical(yy, 4)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

yy=np.around(yy)
print(yy)

print(y_test)

"""# تمرین دوم_شبکه_اول """

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
ds =pd.read_csv('train1.csv')

import pandas as pd
import tensorflow as tf

y=np.array(ds.mobile_wt)
y=y[:1800]
y=np.expand_dims(y,1)
y_max=np.amax(np.array(ds.mobile_wt),axis=0)
y_min=np.amin(np.array(ds.mobile_wt),axis=0)
ya=y_max
yi=y_min
y=(y-y_min)/(y_max-y_min)
y_test=np.array(ds.mobile_wt)
y_test=y_test[-200:]
y_test=np.expand_dims(y_test,1)
ya_t=np.amax(y_test,axis=0)
yi_t=np.amin(y_test,axis=0)

ds_max=np.amax(ds,axis=0)
ds_min=np.amin(ds,axis=0)
dsN=(ds-ds_min)/(ds_max-ds_min)
x=np.array(dsN.drop(['mobile_wt'],axis=1))
x=x[:1800]
print(x)

x_test=np.array(dsN.drop(['mobile_wt'],axis=1))
x_test=x_test[-200:]

model=Sequential()
model.add(Dense(20, activation='relu',input_shape=(20, )))
model.add(Dense(10, activation='relu',input_shape=(20, )))
model.add(Dense(10, activation='relu',input_shape=(10, )))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',  loss='mse')

history=model.fit(x,y,epochs=400);

import matplotlib.pyplot as plt
yy=(model.predict(x_test))*(ya_t-yi_t)+yi_t

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(yy)

print((y_test-yy)*100/y_test)

"""# تمرین دوم_شبکه_دوم """

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from sklearn.cluster import KMeans
ds =pd.read_csv('train1.csv')

import pandas as pd
import tensorflow as tf

y=np.array(ds.mobile_wt)
y=y[:1800]
y=np.expand_dims(y,1)

y_test=np.array(ds.mobile_wt)
y_test=y_test[-200:]
y_test=np.expand_dims(y_test,1)


ds_max=np.amax(ds,axis=0)
ds_min=np.amin(ds,axis=0)
dsN=(ds-ds_min)/(ds_max-ds_min)
x=np.array(dsN.drop(['mobile_wt'],axis=1))
x=x[:1800]
print(x)

x_test=np.array(dsN.drop(['mobile_wt'],axis=1))
x_test=x_test[-200:]

class RBN:
  def __init__(self,x,y,neurons=1500, b=1):
    self.x=x
    self.y=y
    self.neurons=neurons
    self.b=b
    km=KMeans(neurons)
    km.fit(x)
    self.centers = km.cluster_centers_
    self.xx=np.array([[self.gaussian(x[i],self.centers[j],b)for j in range(len(self.centers))] for i in range(len(x)) ])
    self.xx=np.concatenate((np.ones((len(self.xx),1)), self.xx), axis=1)
    self.w2= 0


  def fit(self):
    self.w2=np.linalg.inv(self.xx.T.dot(self.xx)).dot(self.xx.T).dot(self.y)

  def gaussian(self,x,center, b):
      n=np.linalg.norm(x-center)*b
      return np.exp(-n*n)

  def predict(self,x):
    xx=np.array([[self.gaussian(x[i],self.centers[j],self.b)for j in range(len(self.centers))] for i in range(len(x)) ])
    xx=np.concatenate((np.ones((len(xx),1)), xx), axis=1)
    return xx.dot(self.w2)

rbn= RBN(x,y,1500)
rbn.fit()
o=rbn.predict(x_test)

print(100*(np.around(o)-y_test)/y_test)

"""# تمرین دوم_شبکه_سوم """

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
ds =pd.read_csv('train1.csv')

import pandas as pd
import tensorflow as tf

y=np.array(ds.mobile_wt)
y=y[:1800]
y=np.expand_dims(y,1)
y_max=np.amax(np.array(ds.mobile_wt),axis=0)
y_min=np.amin(np.array(ds.mobile_wt),axis=0)
ya=y_max
yi=y_min
y=(y-y_min)/(y_max-y_min)
y_test=np.array(ds.mobile_wt)
y_test=y_test[-200:]
y_test=np.expand_dims(y_test,1)
ya_t=np.amax(y_test,axis=0)
yi_t=np.amin(y_test,axis=0)

ds_max=np.amax(ds,axis=0)
ds_min=np.amin(ds,axis=0)
dsN=(ds-ds_min)/(ds_max-ds_min)
x=np.array(dsN.drop(['mobile_wt'],axis=1))
x=x[:1800]
print(x)

x_test=np.array(dsN.drop(['mobile_wt'],axis=1))
x_test=x_test[-200:]

model=Sequential()
model.add(Dense(60, activation='relu',input_shape=(20, )))
model.add(Dense(60, activation='relu',input_shape=(60, )))
model.add(Dense(60, activation='relu',input_shape=(60, )))
model.add(Dense(60, activation='relu',input_shape=(60, )))
model.add(Dense(60, activation='relu',input_shape=(60, )))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',  loss='mse')

history=model.fit(x,y,epochs=400);

import matplotlib.pyplot as plt
yy=(model.predict(x_test))*(ya_t-yi_t)+yi_t

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(yy)

print((y_test-yy)*100/y_test)