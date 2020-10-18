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
from tensorflow.keras.layers import Input,Activation, Dense,Flatten,Dropout,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D,concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import get_file,get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

## Data directory
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
    # Get the ground truth from generator
    ground_truth = testing_generator.classes

    # Get the label to class mapping from the generator
    label2index = testing_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v,k) for k,v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = model.predict(testing_generator, steps=testing_generator.samples/testing_generator.batch_size,verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)
    
    errors = np.where(predicted_classes != ground_truth)[0]
    accuracy= round((100-((len(errors)/testing_generator.samples)*100)), 2)
    print("No of errors = {}/{}".format(len(errors),testing_generator.samples))
    print('Accuracy : ',accuracy , '%')    
    return accuracy    

def VGG16_Places365(include_top=True, weights='places',
                    input_tensor=None, input_shape=None,
                    pooling=None,
                    classes=365):
    
    ## pretrained VGG16_Places365_weights
    WEIGHTS_PATH = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'

    if not (weights in {'places', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `places` '
                         '(pre-training on Places), '
                         'or the path to the weights file to be loaded.')

    if weights == 'places' and include_top and classes != 365:
        raise ValueError('If using `weights` as places with `include_top`'
                         ' as true, `classes` should be 365')

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten =include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv1')(img_input)

    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool", padding='valid')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv1')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool", padding='valid')(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv1')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv2')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool", padding='valid')(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool", padding='valid')(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool", padding='valid')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)
        
        x = Dense(365, activation='softmax', name="predictions")(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16-places365')

    # load weights
    if weights == 'places':
        if include_top:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

    elif weights is not None:
        model.load_weights(weights)

    return model

def create_places_basic_model(IMAGE_SIZE):
    vgg16_places_model= VGG16_Places365(weights='places', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    vgg16_places_model_out = vgg16_places_model.get_layer('block5_pool').output
    vgg16_places_model_out = GlobalAveragePooling2D()(vgg16_places_model_out)
    vgg16_places_model_out = Dense(8, activation='softmax')(vgg16_places_model_out)
    model = Model(inputs=vgg16_places_model.input,  outputs=vgg16_places_model_out)
    return model

def create_places_mg_model(best_places_basic_model):
    model=load_model(best_places_basic_model)
    Place_model_out = model.get_layer('block1_conv2').output
    Place_model_out = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(Place_model_out)
    branch_1_out = GlobalAveragePooling2D(name='gap1')(Place_model_out)

    Place_model_out = model.get_layer('block2_conv2').output
    Place_model_out = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(Place_model_out)
    branch_2_out = GlobalAveragePooling2D(name='gap2')(Place_model_out)

    Place_model_out = model.get_layer('block3_conv3').output
    Place_model_out = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(Place_model_out)
    branch_3_out = GlobalAveragePooling2D(name='gap3')(Place_model_out)

    Place_model_out = model.get_layer('block4_conv3').output
    Place_model_out = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(Place_model_out)
    branch_4_out = GlobalAveragePooling2D(name='gap4')(Place_model_out)

    Place_model_out = model.get_layer('block5_conv3').output
    Place_model_out = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(Place_model_out)
    branch_5_out = GlobalAveragePooling2D(name='gap5')(Place_model_out)

    merge = concatenate([branch_1_out,branch_2_out,branch_3_out,branch_4_out,branch_5_out])
    output = Dense(8, activation='softmax')(merge)
    model = Model(inputs=model.input, outputs=[output])
    
    return model

### main ----------------------------------------------------------------------------- 

## data generator 
train_generator,validation_generator,testing_generator=data_generator()

## create places_basic_model
places_basic_model=create_places_basic_model(IMAGE_SIZE)

## train places_basic_model
compile_and_train(places_basic_model,"places_basic")

##------------------------------------------------------------------------------------ 
### (change directory to load best_places_basic_model from folder 'training_models')
best_places_basic_model='./pretrained_models/places_basic_model.hdf5'

## create places_mg_model from places_basic_model
places_mg_model=create_places_mg_model(best_places_basic_model)

## train places_mg_model
compile_and_train(places_mg_model,"places_mg")

##------------------------------------------------------------------------------------ 
### (change directory to load best_places_mg_model from folder 'training_models')
best_places_mg_model='./pretrained_models/places_mg_model.hdf5'

## prediction
predict_result(best_places_mg_model,testing_generator)

