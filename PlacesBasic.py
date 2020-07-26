# PlacesBasic <PlacesBasic_model>
#from keras.applications.mobilenet import MobileNet
from vgg16_places_365 import VGG16_Places365
from keras.applications.mobilenet import preprocess_input
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras import models
from keras import layers
from keras.layers import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras import optimizers
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import load_model

#Load the MobileNet model
vgg16_places_model= VGG16_Places365(weights='places', include_top=False, input_shape=(224, 224, 3))

#modify the network
vgg16_places_model_out = vgg16_places_model.get_layer('block5_pool').output
vgg16_places_model_out = GlobalAveragePooling2D()(vgg16_places_model_out)
vgg16_places_model_out = Dense(8, activation='softmax')(vgg16_places_model_out)
model = Model(inputs=vgg16_places_model.input,  outputs=vgg16_places_model_out)

#directory
train_dir = "../train"
validation_dir = "../val"
testing_dir = "../test"

# With Data augmentation 
train_datagen = ImageDataGenerator( 
      horizontal_flip=True,
      vertical_flip=True,
    preprocessing_function = preprocess_input)

validation_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input)

# Change the batchsize according to your system RAM
train_batchsize = 10
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=val_batchsize,
        class_mode='categorical')

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001),
              metrics=['acc'])

# Train the Model
filepath='../G.{epoch:02d}--{val_acc:.2f}.hdf5'
filepath_backup='../R.{epoch:02d}--{val_acc:.2f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=7, mode='auto')
checkpoint=ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpoint_backup=ModelCheckpoint(filepath_backup, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,
      callbacks = [checkpoint,checkpoint_backup,es])

#load trained model
model=load_model('PlacesBasic_model.hdf5')



