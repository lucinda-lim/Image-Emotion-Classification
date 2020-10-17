#!/usr/bin/env python
# coding: utf-8

## Split data into train (80%), val(5%), test(15%)

### requirements:
## places images in 'data' folder, group them by emotion classes 

import numpy as np
import os
import shutil
from math import floor
from sklearn.model_selection import StratifiedShuffleSplit

data_dir = "./data/"

classes_dir = (['amusement/', 'anger/', 'awe/', 'contentment/', 'disgust/', 'excitement/', 'fear/', 'sadness/'])

print("Creating train,test,val folder ...")

for emotion_class in classes_dir:
    
    train_dir = data_dir + 'train/' + emotion_class
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    test_dir = data_dir + 'test/' + emotion_class
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    val_dir = data_dir + 'val/' + emotion_class    
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

#get images into arrays
print("Transferring image names to array...")

folders = os.listdir(data_dir)
data_list=[]
label_list=[]
image_count=0

for folder in folders :
    path = data_dir +folder + '/'
    emotion = os.listdir(path)
    for image in emotion:
        if (os.path.isfile(path+image)):
            data_list.append(image)
            label_list.append(image_count)
    image_count = image_count + 1

data = np.array(data_list)
labels = np.array(label_list)
total = len(data_list)

print("Splitting images to get 15% for testing...")

sss1 = StratifiedShuffleSplit(random_state=18,n_splits=1, test_size=0.15, train_size=0.85)
sss1.get_n_splits(data, labels)

for train_index, test_index in sss1.split(data, labels):
   print("TRAIN:", train_index, "TEST:", test_index)
   X, X_test = data[train_index], data[test_index]
   y, y_test = labels[train_index], labels[test_index]

print("Splitting images to get 5% for validation and 80% for training...")

sss2 = StratifiedShuffleSplit(random_state=18,n_splits=1, test_size=floor(total*0.05), train_size=floor(total*0.80))
sss2.get_n_splits(X, y)

for train_index, test_index in sss2.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_val = X[train_index], X[test_index]
   y_train, y_val = y[train_index], y[test_index]

print("Copying images to new directories...")

for image, label in zip(X_train,y_train):
    train_destination = data_dir  + 'train/'+ classes_dir[label]  + image
    source = data_dir  + classes_dir[label] + image
    shutil.copy(source, train_destination)

print("Images in Training: " + str(len(X_train)))

for image, label in zip(X_test,y_test):
    test_destination = data_dir + 'test/' + classes_dir[label] + image
    source = data_dir  + classes_dir[label] + image
    shutil.copy(source, test_destination)

print("Images in Test: " + str(len(X_test)))

for image, label in zip(X_val,y_val):
    val_destination = data_dir + 'val/'+ classes_dir[label] + image
    source = data_dir  + classes_dir[label] + image
    shutil.copy(source, val_destination)

print("Images in Validation: " + str(len(X_val)))

