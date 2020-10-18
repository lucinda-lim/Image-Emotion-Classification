#!/usr/bin/env python
# coding: utf-8

## combine object and places model with late fusion method

import h5py
import numpy as np
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K,optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input,Activation, Dense,Flatten,Dropout,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D,concatenate,average
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications.mobilenet  import preprocess_input
from tensorflow.keras.utils import get_file,get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

# double data generator
def doubleGenerator(generator1,generator2):
  while True:
    for (x1,y1),(x2,y2) in zip(generator1,generator2):
      yield ([x1,x2],y1)
    
## Settings
BATCH_SIZE=8
IMAGE_SIZE=224
EPOCHS=50
SEED=1

testing_dir = "./data/test"

## Data generator
datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)


test_datagen1 = datagen.flow_from_directory(
    testing_dir,
    target_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = False,
    seed=SEED
    )

test_datagen2 = datagen.flow_from_directory(
    testing_dir,
    target_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = False,
    seed=SEED
    )


##------------------------------------------------------------------------------------ 
##  get layer from object_mg_model (change directory to load best object_mg_model from folder 'training_models')
object_mg_model=load_model('./pretrained_models/object_mg_model.hdf5')

for layer in object_mg_model.layers:
  layer._name = 'object_'+layer.name

object_output = object_mg_model.get_layer(name='object_dense_1').output

##------------------------------------------------------------------------------------ 
## get layer from places_mg_model (change directory to load best places_mg_model from folder 'training_models')
places_mg_model=load_model('./pretrained_models/places_mg_model.hdf5')

for layer in places_mg_model.layers:
  layer._name = 'places_'+layer.name

places_output = places_mg_model.get_layer(name='places_dense_2').output

## Average both models output  
average_layer  = average([object_output, places_output])

## Create late_fusion2_model 
late_fusion2_model = Model(inputs=[Object_MG_Model.input, Places_MG_Model.input], outputs=average_layer)

ground_truth = test_datagen1.classes

## Data generator 
dgenerator=doubleGenerator(test_datagen1,test_datagen2)

##  Predictions
predictions = late_fusion2_model.predict_generator(dgenerator,steps=math.ceil(test_datagen1.samples/BATCH_SIZE), verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
errors = np.where(predicted_classes != ground_truth)[0]
accuracy= round((100-((len(errors)/testing_generator.samples)*100)), 2)
print("No of errors = {}/{}".format(len(errors),testing_generator.samples))
print('Accuracy : ',accuracy , '%')

