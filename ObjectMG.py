# Object_MG_Network "ObjectMG.hdf5"

# libraries
from keras.models import load_model
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate,average
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.applications.mobilenet import preprocess_input
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
import  itertools 
import matplotlib.pyplot as plt
from  sklearn.metrics import confusion_matrix

#directory
train_dir = r"../train"
validation_dir = r"../val"
testing_dir = r"../test"

#load basic object model 
object_model=load_model("../ObjectBasic.hdf5")

# For each branch of the network, add 1x1 2D Convolutional Layer, Global Average Pooling 2D layer
MobileNet_model_out = object_model.get_layer('conv_pw_1_relu').output
conv_1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
branch_1_out = GlobalAveragePooling2D(name='gap1')(conv_1)

MobileNet_model_out = object_model.get_layer('conv_pw_3_relu').output
conv_2 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
branch_2_out = GlobalAveragePooling2D(name='gap2')(conv_2)

MobileNet_model_out = object_model.get_layer('conv_pw_5_relu').output
conv_3 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
branch_3_out = GlobalAveragePooling2D(name='gap3')(conv_3)

MobileNet_model_out = object_model.get_layer('conv_pw_11_relu').output
conv_4 =Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
branch_4_out = GlobalAveragePooling2D(name='gap4')(conv_4)

MobileNet_model_out = object_model.get_layer('conv_pw_13_relu').output
conv_5 = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(MobileNet_model_out)
branch_5_out = GlobalAveragePooling2D(name='gap5')(conv_5)

merge = average([branch_1_out,branch_2_out,branch_3_out,branch_4_out,branch_5_out])
output = Dense(8, activation='softmax')(merge)
model = Model(inputs=object_model.input, outputs=[output])

# Data augmentation 
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
        class_mode='categorical',
        )

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=val_batchsize,
        class_mode='categorical')

#Compile model 
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001),
              metrics=['acc'])

# save model weights directory
filepath='../G.{epoch:02d}--{val_acc:.2f}.hdf5'
filepath_backup='../R.{epoch:02d}--{val_acc:.2f}.hdf5'
es= EarlyStopping(monitor='val_loss', patience=5, mode='auto')
checkpoint=ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpoint_backup=ModelCheckpoint(filepath_backup, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)

#Fit model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,
      callbacks = [checkpoint,checkpoint_backup,es]
)

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


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


# Confusion matrix
def plot_confusion_matrix(cm, classes, figname='cm_default', normalize=True, title='Confusion matrix', cmap=plt.cm.Blues,figsize=(20,20)):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Calculate chart area size
    leftmargin = 0.5 # inches
    rightmargin = 0.5 # inches
    categorysize = 0.5 # inches
    figwidth = leftmargin + rightmargin + (len(classes) * categorysize)      
    f = plt.figure(figsize=(figwidth, figwidth))

    # Create an axes instance and ajust the subplot size
    ax = f.add_subplot(111)
    ax.set_aspect(1)
    f.subplots_adjust(left=leftmargin/figwidth, right=1-(rightmargin/figwidth), top=0.94, bottom=0.1)
    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar(res)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # f.savefig("/confusion_matrix", format='png')
    
cm = confusion_matrix(testing_generator.classes, predicted_classes)
np.set_printoptions(precision=2)
classes=['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
plot_confusion_matrix(cm, classes,  title='Normalized Confusion matrix')
