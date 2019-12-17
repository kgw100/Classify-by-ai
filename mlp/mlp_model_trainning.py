import os, re, glob 
import cv2
import numpy as np 
import shutil 
from numpy import argmax 
from keras.models import load_model 
 
categories = ["h264","hwp","jpg","pdf","png","wav"] 
  
def Dataization(img_path): 
    image_w = 30
    image_h = 110
    img = cv2.imread(img_path) 
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0]) 
    return (img/256) 
  
src = [] 
name = [] 
test = [] 
cnt = 0

image_dir = ''

for file in os.listdir(image_dir): 
    if (file.find('.png') is not -1):       
        src.append(image_dir + file) 
        name.append(file) 
        test.append(Dataization(image_dir + file)) 
        
test = np.array(test) 
model = load_model('c.h5',compile=False) #불러들이는 모델이름
predict = model.predict_classes(test)
predict2 = model.predict(test)

    
for i in range(len(test)): 
    if str(categories[predict[i]]) in name[i].split(".")[0]:
        cnt += 1
    print("File : " + name[i].split(".")[0] + ", Predict : "+ str(categories[predict[i]]) + str(predict2[i]))
    print(cnt)