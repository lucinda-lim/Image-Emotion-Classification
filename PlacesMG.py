
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.layers import concatenate,average
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import EarlyStopping,ModelCheckpoint

#dir
train_dir = r"../train"
validation_dir = r"../val"
testing_dir = r"../test"

#load model for PlacesBasic.py
model=load_model('PlacesBasic_model.hdf5')

#For each branch, add conv2d & Global Average Pooling layer
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


# With Data augmentation 
train_datagen = ImageDataGenerator( 
      horizontal_flip=True,
      vertical_flip=True,
    preprocessing_function = preprocess_input
    )

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

from keras import optimizers
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001),
              metrics=['acc'])

filepath='../G.{epoch:02d}--{val_acc:.2f}.hdf5'
filepath_backup='../R.{epoch:02d}--{val_acc:.2f}.hdf5'

es= EarlyStopping(monitor='val_loss', patience=15, mode='auto')
checkpoint=ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpoint_backup=ModelCheckpoint(filepath_backup, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,
      callbacks = [checkpoint,checkpoint_backup,es]
)


#load_model
model=load_model('PlacesMG_model.hdf5')


# Create a generator for prediction
import numpy as np
testing_generator = validation_datagen.flow_from_directory(
        testing_dir,
        target_size=(224, 224),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

#Get the filenames from the generator
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
print("No of errors = {}/{}".format(len(errors),testing_generator.samples))
print('Accuracy : ', round((100-((len(errors)/testing_generator.samples)*100)), 2), '%')

