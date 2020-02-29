# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:01:58 2020

@author: Rohan Dixit
"""
ep=int(input("enter number of epoch you want to run: "))
print("You enter"+str(ep)+" Epoch")
print("Now Your model create at "+str(ep)+" Epoch")
model=input("Please enter model name which you want to save: ")

from keras.models import Sequential # initialize neural network
from keras.layers import Convolution2D # making CNN to deal with image for video we use 3d
from keras.layers import MaxPooling2D # for proceed poooling step
from keras.layers import Flatten #convert pool to feature 
from keras.layers import Dense #create and connect nn
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 4, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
n=training_set.classes.itemsize
n=n*200
a=training_set.class_indices
li=list(a.keys())
n=int(n)

test_set = test_datagen.flow_from_directory('dataset/test',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
n1=test_set.classes.itemsize
n1=n1*100
from keras.callbacks import EarlyStopping

earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0,patience = 3, verbose = 1,restore_best_weights = True)
callback=[earlystop]
classifier.fit_generator(training_set,callbacks=callback,epochs =ep,validation_data = test_set)

#prediction

classifier.save(model+'.h5')

import matplotlib.pyplot as plt

# from IPython.display import Inline
plt.plot(classifier.history.history['accuracy'])
plt.plot(classifier.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(classifier.history.history['loss'])
plt.plot(classifier.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
