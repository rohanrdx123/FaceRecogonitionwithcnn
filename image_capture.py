# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:27:54 2020

@author: Rohan Dixit
"""
print("This file is for live face Capture for data set ")
print("First  you have to enter the folder or user  name where your training image will save and press enter ")
print("Then you have to enter the same folder or user name name where your testing image will save and press enter ")
print("Now your dataset is ready ")
import cv2
import os
import matplotlib.pyplot as plt
num_of_sample = 500
vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
iter1=0
print("this images for training the data ")
nam  = input('Enter User Name : ')
path23='dataset//train//%s'%(nam)
os.mkdir(path23)
print(path23)
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
        cv2.putText(frame,'face No. '+str(iter1),(x,y), cv2.FONT_ITALIC, 1,(255,0,255),2,cv2.LINE_AA)
        path2 = '%s/%d.png'%(path23,iter1)
        cv2.imwrite(path2,im_f)
        
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    if iter1==500:
        print('Faces captured for training the data data')
        break    
vid.release()
cv2.destroyAllWindows()
# for test data 
print("enter same name as you enter the above it is for test data ")
import cv2
vid1 = cv2.VideoCapture(0)
iter2=0
print("this images for testing the data ")
nam  = input('Enter User Name : ')
path2='dataset//test//%s'%(nam)
n=100
os.mkdir(path2)
print(path2)
while(iter2<n):
    
    r,frame1 = vid1.read();
    frame1 = cv2.resize(frame1,(630,480))
    #im1 = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    face=face_cascade.detectMultiScale(frame1)
    for x,y,w,h in (face):
        cv2.rectangle(frame1,(x,y),(x+w,y+h),[255,0,0],4)
        iter2=iter2+1
        im_f = frame1[y:y+h,x:x+w]
        im_f = cv2.resize(im_f,(112,92))
        cv2.putText(frame1,'face No. '+str(iter2),(x,y), cv2.FONT_ITALIC, 1,(255,0,255),2,cv2.LINE_AA)
        path21 = '%s/%d.png'%(path2,iter2)
        cv2.imwrite(path21,im_f)
        
    cv2.imshow('frame',frame1)
    cv2.waitKey(1)
    if iter==100:
        print('Faces captured  for testing the data')
        break  
print("Now you have to run the second file (train_model.py) file ")
vid1.release()
cv2.destroyAllWindows()
