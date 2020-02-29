# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:27:54 2020

@author: Rohan Dixit
"""
print("This file is for live face detection")
print("First you have to load the model then press enter ")
print("Then you have to enter the folder name where your test will save and press enter ")
print("Now your live face detection is ready to detect ")

import cv2
import os
from tensorflow.keras.models import load_model
#load the haarcascade file for face recogonition
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#load the model 
model=input("Please enter model name which you want to load: ")
classifier = load_model(model+'.h5')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/train',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
a=training_set.class_indices
li=list(a.keys())

num_of_sample = 1000
vid = cv2.VideoCapture(0)

iter1=0

nam  = input('Enter folder name where your test images will save : ')
p='testimage//%s'%(nam)
os.mkdir(p)
print(p)
from tensorflow.keras.preprocessing import image
import numpy as np
while(iter1<num_of_sample):
    
    r,frame = vid.read();
    frame = cv2.resize(frame,(640,480))
    #im1 = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    face=face_cascade.detectMultiScale(frame)
    for x,y,w,h in (face):
        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],4)
        iter1=iter1+1
        im_f = frame[y:y+h,x:x+w]
        im_f = cv2.resize(im_f,(112,92))
        p2 = '%s/%d.png'%(p,iter1)
        cv2.imwrite(p2,im_f)
        im_f=p2
        img2 = image.load_img(im_f, target_size=(64, 64))
        img = image.img_to_array(img2)
        img = img/255
        img = np.expand_dims(img, axis=0)
        prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
        s1=prediction[0,0]*100
        s2=prediction[0,1]*100
        s3=prediction[0,2]*100
        s4=prediction[0,3]*100
        v=print("i am "+str(s1)+" % sure about"+li[0])
        v=print("i am "+str(s2)+" % sure about"+li[1])
        v=print("i am "+str(s3)+" % sure about"+li[2])
        v=print("i am "+str(s4)+" % sure about"+li[3])
        if(prediction[0,0]*100)>20.00:
            cv2.putText(frame,str(li[0]),(x,y), cv2.FONT_ITALIC, 1,(255,0,255),2,cv2.LINE_AA)
        elif (prediction[0,1]*100)>20.00:
            cv2.putText(frame,str(li[1]),(x,y), cv2.FONT_ITALIC, 1,(255,0,255),2,cv2.LINE_AA)
        elif (prediction[0,2]*100)>20.00:
            cv2.putText(frame,str(li[2]),(x,y), cv2.FONT_ITALIC, 1,(255,0,255),2,cv2.LINE_AA)
        elif (prediction[0,3]*100)>20.00:
            cv2.putText(frame,str(li[3]),(x,y), cv2.FONT_ITALIC, 1,(255,0,255),2,cv2.LINE_AA)
        else:
            cv2.putText(frame,'None '+str(iter1),(x,y), cv2.FONT_ITALIC, 1,
                    (255,0,255),2,cv2.LINE_AA)
        
    cv2.imshow('Video',frame) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

vid.release()
cv2.destroyAllWindows()

    
    
    
    

