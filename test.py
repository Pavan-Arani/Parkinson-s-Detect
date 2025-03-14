#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# Download and extract the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir) / 'flower_photos' 

# Count the number of images in the corrected dataset path
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Image count:", image_count)

