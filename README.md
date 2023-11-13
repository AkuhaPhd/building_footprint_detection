# Satellite Image Segmentation Module

This Python module provides functionalities for segmenting objects in satellite or aerial images, creating masks, and extracting polygons. It utilizes the Segment Anything (SAM) model for segmentation and includes preprocessing and post-processing steps.

## Installation
- Open notebook in `notebook`in google colab 
- Cd into root dir of main.p
- ensure colab is running on GPU
- Run cells


## TODO Damage Assessment:
### 1. Obtain or Generate Tagged Images:
Acquire or generate tagged images of damaged buildings. This dataset will be used for training and evaluating the damage assessment model.

### 2. Model Training:
Fine-tune or train a separate model (e.g., a classifier) for damage assessment using the tagged images. Classify buildings into categories such as 'none,' 'low,' 'medium,' 'high,' or 'full' damage.

### 3. Damage Assessment:
Assess the degree of damage for each building, categorizing it on the specified scale. This could be achieved by applying the trained damage assessment model to the detected building footprints.
