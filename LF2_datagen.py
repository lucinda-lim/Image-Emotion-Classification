import h5py
import numpy as np
from keras.models import Model
from keras.layers import multiply,average,Dense,concatenate
import keras
import math
import tensorflow
from keras import initializers,regularizers,constraints
from keras import backend as K 
from keras.layers import multiply,average,Dense
from keras.models import Model
from keras.models import load_model

# object 
object_model = keras.models.load_model('../ObjectMG_model.hdf5')
for layer in object_model.layers:
  layer.name = 'object_'+layer.name
object_output = object_model.get_layer(name='object_dense_1').output

# places 
places_model = keras.models.load_model('../PlacesMG_model.hdf5')
for layer in places_model.layers:
  layer.name = 'places_'+layer.name
places_output = places_model.get_layer(name='places_dense_2').output

#doubleGenerator
def doubleGenerator(generator1,generator2):
  while True:
    for (x1,y1),(x2,y2) in zip(generator1,generator2):
      yield ([x1,x2],y1)

#preprocess image
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = preprocess_input
)


test_datagen1 = datagen.flow_from_directory(
    '../test',
    target_size = (224,224),
    batch_size = 8,
    shuffle = False,
    seed=1
    )

test_datagen2 = datagen.flow_from_directory(
    '../test',
    target_size = (224,224),
    batch_size = 8,
    shuffle = False,
    seed=1
    )


ground_truth = test_datagen1.classes
dgenerator=doubleGenerator(test_datagen1,test_datagen2)
predictions = model_2.predict_generator(dgenerator,steps=math.ceil(3339/8), verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),test_datagen1.samples))
print('Accuracy : ', round((100-((len(errors)/test_datagen1.samples)*100)), 2), '%')

model_2.save("../LF2_datagen.h5")

