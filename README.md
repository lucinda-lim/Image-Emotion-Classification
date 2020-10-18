# Where is the emotion? Dissecting a multi-gap network for image emotion classification 
Code written in *Tensorflow.keras*

## Description
Classify emotion into 8 emotion categories.

## Architecture
<INSERT MG network>
Details at *research_paper.pdf*

## Late Fusion Method
<INSERT Late Fusion image>
**Results**
<insert result table /confusion matrix>

## Train
1. Download [FI dataset] (https://www.cs.rochester.edu/u/qyou/deepemotion/) 
2. Store images in `data` folder groupby emotion classes.
2. Split data with `'split_data.py'`
3. Run `object.py` and `places.py` (Change directory to `training_models` folder)
4. Run `late_fusion2.py` (Switch directory to `training_models` folder)

## Use Pretrained Models 
1. Download [FI pretrained weights](https://drive.google.com/drive/folders/1Gm5fyY8bthkENOsTxR9oe08r15wc7vyV?usp=sharing), Store them in `pretrained_models` folder
2. Store test images in 'data/test' folder.
3. Run `late_fusion2.py`

## Demo App with Streamlit
1. `!pip install streamlit`
2. Run `demo.py` script with command `streamlit run demo.py`

## Acknowledgement
Part of this code is borrowed from [GKalliatakis](https://github.com/GKalliatakis/Keras-VGG16-places365) respository.
