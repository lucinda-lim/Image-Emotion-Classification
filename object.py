#!/usr/bin/env python
# coding: utf-8

import os
import glob 
import pandas as pd
import numpy as np

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
from tensorflow.keras.applications import MobileNet

## Data Directory
TRAIN_DIR = "./data/train/"
VAL_DIR = "./data/val"
TEST_DIR = "./data/test"

## Settings 
BATCH_SIZE=10
IMAGE_SIZE=224
EPOCHS=50

## Functions 
def data_generator():

    train_datagen = ImageDataGenerator( 
          horizontal_flip=True,
          vertical_flip=True,
        preprocessing_function = preprocess_input)

    validation_datagen = ImageDataGenerator(
            preprocessing_function = preprocess_input)

    train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    testing_generator = validation_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False)
    
    return train_generator,validation_generator,testing_generator

def compile_and_train(model,MODEL_NAME):
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001),
                  metrics=['acc'])
    
    TRAINED_MODEL_PATH='./training_models/'+MODEL_NAME+'.{epoch:02d}--{val_acc:.2f}.hdf5'
    early_stop= EarlyStopping(monitor='val_loss', patience=7, mode='auto')
    checkpoint=ModelCheckpoint(TRAINED_MODEL_PATH, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)  
    
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=train_generator.samples/train_generator.batch_size ,
          epochs=EPOCHS,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples/validation_generator.batch_size,
          verbose=1,
          callbacks = [checkpoint,early_stop])
    
def predict_result(model,testing_generator):
    
    model=load_model(model)
    fnames = testing_generator.filenames
    ground_truth = testing_generator.classes
    label2index = testing_generator.class_indices
    idx2label = dict((v,k) for k,v in label2index.items())
  
    predictions = model.predict(testing_generator, steps=testing_generator.samples/testing_generator.batch_size,verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)
    
    errors = np.where(predicted_classes != ground_truth)[0]
    accuracy= round((100-((len(errors)/testing_generator.samples)*100)), 2)
    print("No of errors = {}/{}".format(len(errors),testing_generator.samples))
    print('Accuracy : ',accuracy , '%')
    
    return accuracy    

def create_object_basic_model():
    
    MobileNet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    MobileNet_model_out = MobileNet_model.get_layer('conv_pw_13_relu').output
    MobileNet_model_out= GlobalAveragePooling2D()(MobileNet_model_out)
    MobileNet_model_out = Dense(8, activation='softmax')(MobileNet_model_out)
    model = Model(inputs=MobileNet_model.input, outputs=MobileNet_model_out)
    
    return model 

def create_object_mg_model(best_object_basic_model):
    
    model=load_model(best_object_basic_model)
    MobileNet_model_out = model.get_layer('conv_pw_1_relu').output
    conv_1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
    branch_1_out = GlobalAveragePooling2D(name='gap1')(conv_1)

    MobileNet_model_out = model.get_layer('conv_pw_3_relu').output
    conv_2 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
    branch_2_out = GlobalAveragePooling2D(name='gap2')(conv_2)

    MobileNet_model_out = model.get_layer('conv_pw_5_relu').output
    conv_3 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
    branch_3_out = GlobalAveragePooling2D(name='gap3')(conv_3)

    MobileNet_model_out = model.get_layer('conv_pw_11_relu').output
    conv_4 =Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
    branch_4_out = GlobalAveragePooling2D(name='gap4')(conv_4)

    MobileNet_model_out = model.get_layer('conv_pw_13_relu').output
    conv_5 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
    branch_5_out = GlobalAveragePooling2D(name='gap5')(conv_5)

    merge = average([branch_1_out,branch_2_out,branch_3_out,branch_4_out,branch_5_out])
    output = Dense(8, activation='softmax')(merge)
    model = Model(inputs=model.input, outputs=[output])
    
    return model 

### main -----------------------------------------------------------------------------

## data generator
train_generator,validation_generator,testing_generator=data_generator()

## create object_basic_model
object_basic_model=create_object_basic_model()

## train object_basic_model
compile_and_train(object_basic_model,"object_basic")

##---------------------------------------------------------------------------------------
### (change directory to load best object-basic model from folder 'training_models')
best_object_basic_model='./pretrained_models/object_basic_model.hdf5'

## create object_mg_model from object_basic_model
object_mg_model=create_object_mg_model(best_object_basic_model)

## data generator
train_generator,validation_generator,testing_generator=data_generator()

## train object_mg_model 
compile_and_train(object_mg_model,"object_mg")

##---------------------------------------------------------------------------------------
### (change directory to load best object-mg model from folder 'training_models')
best_object_mg_model='./pretrained_models/object_mg_model.hdf5'

## prediction
predict_result(best_object_mg_model,testing_generator)

