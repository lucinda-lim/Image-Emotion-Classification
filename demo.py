## late_fusion2_model Demo with Streamlit App

### requirements:
## pip install streamlit

### run command: 
##  streamlit run demo.py

import streamlit as st 
import pandas as pd
import numpy as np
from PIL import Image,ImageOps

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet  import preprocess_input

## Surpress warning
st.set_option('deprecation.showfileUploaderEncoding', False)

## Upload image interface
st.title("Image Emotion Model")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

## Load pretrained model 
emotion_model = tf.keras.models.load_model('./pretrained_models/late_fusion2_model.h5')

## After image uploaded 
if uploaded_file is not None:
    
    ## Image preprocessing
    image = Image.open(uploaded_file) 
    size = (224,224) 
    im = ImageOps.fit(image, size, Image.ANTIALIAS)
    im = img_to_array(im)
    im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    im = preprocess_input(im)
    
    ## Predictions
    preds = emotion_model.predict([im,im])
    predicted_index=np.argmax(preds,axis=1)[0] 
    labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust','excitement', 'fear', 'sadness']
    predicted_classes=str(labels[predicted_index])
    
    ## Display 
    st.image(image,width=400,height=400)
    st.subheader('Predicted Emotion: '+predicted_classes)    
    st.subheader('Emotion Distribution:')
    st.write(pd.DataFrame({'emotion_classes':labels\
                           ,'predicted_probability':preds[0]}))



