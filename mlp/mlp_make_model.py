from keras.models import Sequential 
from keras.layers import Dropout, Activation, Dense 
from keras.layers import Flatten, Convolution2D, MaxPooling2D 
from keras.models import load_model 
import numpy as np 
import cv2 

categories = ["h264","hwp","jpg","pdf","png","wav"] #파일 조각 식별 카테고리 

(ImageData1217_2 이면 h264 빼는식으로) 설정
num_classes = len(categories)
X_train, X_test, Y_train, Y_test = np.load('/project/model/ImageData1217_1.npy', allow_pickle = True) 
model = Sequential() 
print(X_train.shape[1:])
model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', input_shape=(110,30,3)))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Convolution2D(64, 3, 3,  activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Convolution2D(64, 3, 3)) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(256, activation = 'relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(num_classes,activation = 'softmax'))
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy']) 
[model.fit](http://model.fit/)(X_train, Y_train, batch_size=128, nb_epoch=10) 
score = model.evaluate(X_test, Y_test)  
print('loss==>' ,score[0]*100)  
print('accuracy==>', score[1]*100)
[model.save](http://model.save/)('/project/model/CNNMode1217_1.h5') #아웃풋 모델 