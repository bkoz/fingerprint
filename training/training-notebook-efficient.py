#!/usr/bin/env python
# coding: utf-8

# # Notebook Overview
# 
# In this notebook, you will decompress the trainig data and extract the data labels (M, F, Left, Right, Thumb, Index, Middle, Ring, Little) from the image filenames. You will create a training, validation and test dataset along with a sequential model graph. Lastly you will train, score and save the model off for later use.
# 
# 1. Setup the software requirements
# 1. Prepare the image for training
# 1. Define, compile and train a model
# 1. Test the trained model and score it's accuracy

# # Setup

# ## Install Requirements

# In[3]:


get_ipython().system('pip install -r ../../../requirements.txt -q')


# ## Import modules 

# In[4]:


# used for the datasets
import numpy as np
import pandas as pd
# to plot the data
import seaborn as sns
import tensorflow as tf

import os
import cv2
import random

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split


# # Data Prep

# ## Decompress training data

# In[3]:


get_ipython().system('tar -xJf ../compressed_data/fingerprint_altered_easy.tar.xz -C ../data')


# ## View a sample of the data

# In[4]:


# Importing image data into Numpy arrays
img = mpimg.imread('../data/fingerprint_altered_easy/1__M_Left_index_finger_CR.BMP')
print(img)


# In[5]:


# Plotting numpy arrays as images
imgplot = plt.imshow(img)


# In[6]:


# resize the image to 96x96
# img.thumbnail((96, 96)) #resizes the image


# In[7]:


# Plotting numpy arrays as images
imgplot = plt.imshow(img, cmap="gray")


# ## Create function to extract labels from file names

# In[5]:


def extract_label(img_path,train = True):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    subject_id, etc = filename.split('__')
    
    if train:
        gender, lr, finger, _, _ = etc.split('_')
    else:
        gender, lr, finger, _ = etc.split('_')
    
    gender = 0 if gender == 'M' else 1
    lr = 0 if lr == 'Left' else 1

    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
    
    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)


# ## Create a function to load and resize images

# In[6]:


img_size = 96


def loading_data(path, train):
    print("loading data from: ", path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            label = extract_label(os.path.join(path, img),train)
            #print(f'label = {label}')
            data.append([label[2], img_resize ])
        except Exception as e:
            pass
    data
    return data


# ## Configure the data paths

# In[7]:


Easy_path = "../data/fingerprint_altered_easy"
Easy_data = loading_data(Easy_path, train = True)

data = np.concatenate([Easy_data], axis=0, dtype=object)


# ## Create a complete array with label and feature

# In[8]:


X, y = [], []

for label, feature in data:
    X.append(feature)
    y.append(label)
    
#del data

X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X / 255.0

y = np.array(y)


# In[9]:


# print the first record in the new array
print(data[0])


# ## Split the data into train and validation

# In[10]:


#train_test_split function helps us create our training data and test data from a single dataset
X_train, X_test, y_train, y_test = train_test_split(
    # input data
    X, 
    # target data
    y, 
    # size of the test dataset
    test_size=0.2, 
    # a "seed" that initializes the psuedorandom number generator
    random_state=42, 
    # controls whether the input dataset is randomly shuffled before being split
    shuffle=True, 
    # controls if the data are split in a stratified fashion
    stratify=None
)

# The output objects are Numpy arrays.


# ## Describe the arrays

# In[11]:


print("Full dataset:  ",X.shape)
#del X, y
print("Train split:      ",X_train.shape)
print("Traing target:       ",y_train.shape)
print("Test split:      ",X_test.shape)
print("Test target:       ",y_test.shape)


# # Modeling
# 
# Model Training APIs
# 1. Define the model architecture
# 1. Complile the model with the .compile API, which onfigures the model for training
# 1. Train or Fit the model with the .fit API, which trains the model for a fixed number of epochs (iterations on a dataset).
# 1. Predict with the model

# ## Define the model

# In[12]:


# create a Sequential model
model = Sequential(name="fingerprint_prediction")

# incrementally create layers using the .add method
# Optionally, the first layer can receive an `input_shape` argument:
# start with the input shape equal to the fingerprint image size of 96x96x1
# we learn a total of 32 filters. 
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (96,96,1), name="layer1"))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', name="layer2"))
# Max pooling is then used to reduce the spatial dimensions of the output volume.
model.add(MaxPool2D(pool_size=(2,2)))
# Dropout???s purpose is to help your network generalize and not overfit.
model.add(Dropout(0.25))

# We then learn 64 filters. Again max pooling is used to reduce the spatial dimensions.
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', name="layer3"))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu', name="layer4"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))

model.summary()


# ## Compile the model
# 
# First, we want to decide a model architecture, this is the number of hidden layers and activation functions, etc. (compile)

# In[13]:


model_path = './model'

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None
)

# Callbacks: utilities called at certain points during model training.
callbacks = [
    # Stop training when a monitored metric has stopped improving.
    EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1),
    # Callback to save the Keras model or model weights at some frequency.
    ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max', verbose=1),
    # Reduce learning rate when a metric has stopped improving.
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
]


# ## Train the model
# 
# Secondly, we will want to train our model to get all the paramters to the correct value to map our inputs to our outputs. (fit)

# In[17]:


# In Keras, callback records events into a History object.

epochs = 20
batch_size = 32
multi_proc = True

model.fit(
    # Input data
    x=X_train,
    # Target data
    y=y_train,
    # Number of samples per gradient update. 
    batch_size=batch_size,
    # Number of iterations to train the model. 
    epochs=epochs,
    # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    verbose="auto",
    # This callback is automatically applied to every Keras model, then the History object gets returned by the fit method of models.
    callbacks=None,
    # Float between 0 and 1. Fraction of the training data to be used as validation data. 
    validation_split=0.1,
    # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. Will override val_split
    validation_data=None,
    # whether to shuffle the training data before each epoch
    shuffle=True,
    # Optional dictionary mapping class indices (integers) to a weight (float) value
    class_weight=None,
    # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
    sample_weight=None,
    # Epoch at which to start training (useful for resuming a previous training run).
    initial_epoch=0,
    # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
    steps_per_epoch=None,
    # Only relevant if validation_data is provided and is a tf.data dataset.
    validation_steps=None,
    # Number of samples per validation batch. If unspecified, will default to batch_size.
    validation_batch_size=None,
    # Only relevant if validation data is provided.
    validation_freq=1,
    # Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
    max_queue_size=10,
    # Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
    workers=4,
    # If True, use process-based threading. If unspecified, use_multiprocessing will default to False. 
    use_multiprocessing=multi_proc,
)

#TODO training can take up to 5 minutes per epoch on an AWS m5a.4xlarge, peaking around 16GiB and 10CPU


# ## Evaluate the model
# 
# Returns the loss value & metrics values for the model in test mode.

# In[14]:


type(X_test)
X_test[0].shape


# In[19]:


batch_size = None
workers = 1
multi_proc = False

score = model.evaluate(
    # Input data. 
    x=X_test,
    # Target data.
    y=y_test,
    # Number of samples per batch of computation. 
    batch_size=batch_size,
    # "auto", 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
    verbose="auto",
    # Optional Numpy array of weights for the test samples, used for weighting the loss function.
    sample_weight=None,
    # Total number of steps (batches of samples) before declaring the evaluation round finished.
    steps=None,
    # List of callbacks to apply during evaluation.
    callbacks=None,
    # Maximum size for the generator queue. 
    max_queue_size=10,
    # Maximum number of processes to spin up when using process-based threading. Default is 1.
    workers=workers,
    # use process-based threading
    use_multiprocessing=False,
    # If True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.
    return_dict=False
)

print("Test loss:", score[0])
print("Test accuracy:", score[1])


# # Save the model
# 
# There are two formats you can use to save an entire model to disk: the TensorFlow SavedModel format, and the older Keras H5 format.

# In[20]:


# older Keras H5 format
model.save('model.h5')


# In[21]:


# TensorFlow SavedModel format
model.save('model')


# ## Cleanup 

# In[22]:


# !rm -rf ../data/*


# In[23]:


# !rm -rf .ipynb_checkpoints/


# # Predict against data
# 
# Lastly, we will want to use this model to do some feed-forward passes to predict novel inputs. (predict). Generates output predictions for the input samples.

# ## Load the trained model

# In[24]:


fingerprint_model = load_model('model.h5')


# In[15]:


fingerprint_model = load_model('model')


# ## Decompress the real data

# In[26]:


# !tar -xJf ../compressed_data/fingerprint_real.tar.xz -C ../data


# In[27]:


# img = mpimg.imread('../data/fingerprint_altered_easy/1__M_Left_index_finger_CR.BMP')

img_array = cv2.imread('../data/fingerprint_real/103__F_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(img_array, (img_size, img_size))

imgplot = plt.imshow(img_resize)


# In[28]:


image_index = 0
print(img_resize.shape)
print(type(img_resize))
# print(img_resize)
img_resize.resize(1, 96, 96, 1)
np.random.rand(3,2)
myimg = np.random.rand(96, 96, 1)
myimg = np.random.randint(255, size=(1, 96, 96, 1), dtype=int)
print(myimg.shape)
print(type(myimg))
# print(myimg)

print(type(X_test))
print(X_test[0:1].shape)
# print(X_test[0:1])


# In[16]:


#
# Load the real dataset.
#
Real_path = "../data/fingerprint_real"
Real_data = loading_data(Real_path, train = False)

real_data = np.concatenate([Real_data], axis=0, dtype=object)

X, y = [], []

#
# Separate the labels and features.
#
for label, feature in data:
    X.append(feature)
    y.append(label)

#
# Convert the real list to numpy then reshape and normalize.
#
X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X / 255.0

#
# Convert the labels list to a numpy array.
#
y = np.array(y)

print(f'Real image count = {data.shape[0]}')
print(f'Real array shape = {X.shape}')


# In[17]:


prediction = fingerprint_model.predict(
    # Input samples. 
    x = X[20:25],
    batch_size=None,
    verbose="auto",
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)

np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
print(f'Real dataset labels = {y[20:25]}')
print(f'Real dataset predictions = {prediction}')


# In[18]:


def single_prediction(img_array):
    prediction = fingerprint_model.predict(
        x=img_array,
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    return prediction


# In[19]:


#
# Make a prediction from a single image file.
#
F_Left_index = cv2.imread('../data/fingerprint_real/103__F_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(F_Left_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
print(f'Prediction = {single_prediction(img_resize)}')

F_Left_index = cv2.imread('../data/fingerprint_real/275__F_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(F_Left_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
print(f'Prediction = {single_prediction(img_resize)}')

M_Right_index = cv2.imread('../data/fingerprint_real/232__M_Right_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(M_Right_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
print(f'Prediction = {single_prediction(img_resize)}')

M_Right_index = cv2.imread('../data/fingerprint_real/504__M_Right_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(M_Right_index, (img_size, img_size))
img_resize.resize(1, 96, 96, 1)
print(f'Prediction = {single_prediction(img_resize)}')


# ### Create a request json file from the last image.

# In[38]:


import json
import numpy
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

encodedNumpyData = json.dumps(img_resize, cls=NumpyArrayEncoder)

print("serialize NumPy array into JSON and write into a file")
with open("fingerprint.json", "w") as write_file:
    json.dump(img_resize, write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")


# In[45]:


req = {
    "inputs": [
      {
        "name": "layer1_input",
        "shape": [1, 96, 96, 1],
        "datatype": "FP32",
        "data": [[[[160], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [158], [0], [0], [0], [0]], [[160], [105], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [121], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [254], [255], [255], [255], [255], [255], [255], [255], [246], [249], [255], [246], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [243], [254], [255], [255], [255], [253], [255], [255], [239], [255], [255], [255], [255], [233], [229], [244], [233], [253], [246], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [254], [255], [255], [255], [255], [255], [253], [253], [254], [233], [251], [237], [233], [241], [255], [255], [255], [225], [249], [237], [249], [228], [255], [236], [236], [255], [251], [205], [217], [243], [255], [255], [255], [255], [254], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [234], [243], [245], [251], [247], [255], [251], [231], [227], [202], [236], [240], [254], [255], [243], [209], [192], [163], [242], [226], [237], [255], [255], [211], [152], [217], [249], [254], [255], [255], [242], [253], [241], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [235], [243], [239], [197], [177], [236], [206], [253], [255], [221], [221], [210], [172], [236], [251], [232], [203], [175], [199], [174], [201], [207], [253], [255], [204], [191], [231], [254], [253], [255], [243], [252], [238], [251], [247], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [250], [247], [206], [255], [255], [250], [188], [242], [215], [205], [209], [134], [209], [254], [247], [244], [234], [198], [204], [149], [186], [219], [236], [225], [212], [109], [94], [163], [214], [255], [225], [230], [221], [167], [205], [250], [255], [252], [252], [216], [219], [251], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [252], [228], [150], [255], [255], [252], [222], [200], [255], [255], [239], [212], [233], [165], [187], [174], [252], [255], [233], [78], [77], [79], [190], [243], [252], [195], [98], [75], [74], [194], [247], [249], [236], [166], [138], [165], [250], [255], [255], [238], [237], [251], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [247], [227], [230], [255], [255], [232], [83], [142], [246], [247], [255], [255], [224], [151], [32], [97], [233], [254], [213], [150], [65], [52], [62], [171], [254], [223], [189], [79], [224], [60], [216], [254], [239], [149], [105], [153], [197], [218], [224], [255], [254], [255], [255], [251], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [231], [210], [255], [255], [255], [255], [250], [255], [237], [250], [81], [78], [54], [96], [143], [255], [255], [238], [213], [98], [37], [180], [196], [243], [213], [99], [5], [81], [42], [251], [251], [229], [236], [61], [154], [184], [255], [239], [199], [148], [97], [220], [119], [153], [244], [255], [255], [254], [251], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [254], [255], [246], [134], [155], [249], [85], [233], [248], [254], [253], [255], [240], [239], [198], [60], [206], [84], [174], [126], [249], [232], [183], [57], [17], [122], [212], [242], [225], [54], [0], [48], [33], [206], [253], [222], [39], [16], [123], [207], [252], [182], [64], [249], [241], [139], [132], [217], [255], [255], [254], [255], [252], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [188], [161], [123], [114], [76], [253], [172], [229], [78], [37], [22], [44], [128], [167], [236], [254], [252], [228], [96], [52], [3], [51], [243], [254], [231], [26], [5], [3], [87], [250], [252], [188], [1], [0], [3], [227], [255], [245], [46], [2], [8], [114], [254], [53], [116], [255], [253], [196], [122], [157], [253], [255], [255], [253], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [246], [255], [247], [246], [255], [252], [249], [244], [228], [141], [59], [17], [36], [119], [2], [60], [213], [241], [254], [180], [45], [18], [5], [205], [240], [249], [163], [32], [10], [20], [230], [245], [249], [87], [9], [46], [230], [239], [253], [138], [23], [45], [147], [253], [11], [43], [234], [253], [255], [255], [91], [185], [223], [255], [204], [220], [254], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [253], [249], [237], [238], [247], [130], [139], [255], [194], [163], [90], [255], [245], [255], [209], [217], [145], [63], [45], [35], [13], [208], [222], [244], [105], [47], [105], [2], [138], [231], [247], [173], [81], [16], [65], [207], [242], [174], [33], [18], [0], [187], [240], [123], [24], [7], [188], [201], [24], [3], [199], [245], [255], [97], [55], [89], [234], [255], [218], [248], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [243], [238], [228], [194], [107], [99], [167], [208], [255], [221], [144], [224], [209], [249], [191], [255], [187], [246], [228], [224], [202], [217], [37], [1], [2], [137], [206], [246], [145], [35], [9], [56], [156], [205], [216], [100], [34], [2], [121], [212], [250], [100], [41], [0], [111], [255], [200], [26], [1], [195], [193], [52], [11], [144], [244], [239], [93], [57], [91], [236], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [248], [208], [186], [212], [164], [117], [74], [89], [89], [237], [98], [49], [76], [99], [162], [163], [233], [188], [222], [255], [145], [69], [237], [202], [72], [26], [0], [11], [154], [190], [159], [101], [71], [25], [7], [122], [202], [180], [75], [38], [51], [192], [251], [217], [40], [8], [218], [250], [162], [70], [28], [170], [241], [105], [25], [147], [244], [196], [65], [59], [136], [209], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [239], [241], [236], [255], [255], [250], [247], [242], [255], [214], [168], [95], [86], [53], [52], [2], [115], [121], [165], [181], [206], [210], [148], [255], [252], [151], [86], [7], [0], [20], [84], [166], [178], [148], [90], [67], [42], [138], [189], [179], [116], [19], [179], [191], [178], [97], [116], [171], [207], [250], [136], [53], [132], [182], [147], [13], [139], [190], [220], [108], [62], [75], [209], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [249], [221], [172], [156], [145], [130], [151], [167], [232], [246], [206], [187], [194], [110], [108], [80], [54], [48], [66], [106], [42], [197], [189], [215], [248], [120], [78], [42], [0], [0], [34], [160], [175], [215], [107], [72], [40], [166], [224], [103], [48], [42], [141], [209], [169], [81], [68], [154], [251], [193], [62], [27], [175], [124], [71], [55], [154], [245], [130], [40], [110], [255], [255], [254], [252], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [246], [221], [221], [141], [106], [29], [31], [10], [3], [16], [69], [227], [128], [77], [119], [129], [129], [208], [186], [179], [152], [117], [56], [0], [60], [50], [81], [128], [141], [178], [160], [126], [80], [25], [16], [33], [124], [193], [212], [112], [22], [127], [204], [140], [33], [5], [74], [223], [189], [103], [9], [124], [222], [200], [57], [33], [153], [186], [87], [9], [155], [223], [140], [65], [169], [255], [254], [252], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [254], [203], [201], [208], [163], [196], [136], [128], [138], [154], [150], [147], [156], [185], [240], [40], [5], [0], [1], [1], [69], [79], [100], [110], [173], [186], [145], [143], [102], [18], [0], [11], [74], [109], [149], [205], [163], [86], [0], [0], [57], [110], [212], [139], [46], [73], [175], [173], [83], [0], [84], [242], [235], [132], [11], [123], [159], [193], [26], [24], [88], [180], [114], [23], [90], [253], [53], [43], [232], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [249], [171], [173], [210], [230], [215], [236], [239], [234], [251], [255], [255], [255], [255], [255], [246], [183], [166], [163], [159], [144], [101], [25], [0], [0], [45], [91], [115], [165], [222], [167], [142], [140], [13], [3], [25], [86], [160], [174], [142], [94], [79], [16], [73], [106], [152], [152], [60], [106], [202], [155], [5], [85], [186], [247], [169], [67], [36], [111], [172], [118], [3], [92], [215], [0], [5], [151], [187], [113], [99], [248], [255], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [184], [245], [178], [232], [250], [255], [255], [234], [243], [156], [85], [103], [118], [84], [73], [75], [82], [92], [83], [86], [180], [127], [198], [193], [178], [113], [2], [151], [10], [34], [71], [70], [79], [127], [143], [183], [163], [30], [32], [54], [131], [218], [77], [96], [167], [133], [55], [71], [56], [20], [71], [251], [163], [1], [101], [84], [229], [200], [37], [9], [69], [60], [75], [79], [93], [106], [0], [28], [209], [230], [181], [111], [254], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [234], [253], [243], [255], [254], [160], [74], [48], [52], [101], [191], [38], [39], [3], [31], [1], [3], [6], [3], [4], [33], [19], [46], [54], [76], [108], [200], [244], [57], [0], [0], [0], [5], [19], [41], [76], [117], [163], [144], [7], [20], [67], [182], [30], [51], [144], [180], [64], [14], [9], [1], [226], [249], [192], [24], [32], [47], [213], [209], [35], [0], [2], [22], [112], [206], [227], [46], [0], [49], [145], [252], [212], [196], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [255], [166], [119], [103], [20], [4], [68], [161], [182], [209], [209], [204], [211], [225], [204], [217], [212], [92], [0], [0], [0], [0], [0], [4], [13], [36], [126], [216], [217], [191], [150], [74], [33], [23], [4], [12], [28], [93], [220], [72], [3], [35], [196], [107], [19], [47], [171], [207], [31], [186], [153], [93], [253], [194], [5], [0], [28], [79], [212], [70], [0], [0], [16], [40], [233], [227], [94], [2], [16], [59], [255], [244], [255], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [255], [233], [108], [72], [119], [10], [46], [51], [191], [243], [250], [251], [251], [254], [252], [239], [226], [218], [181], [132], [66], [227], [211], [150], [21], [0], [0], [0], [0], [7], [16], [21], [148], [249], [39], [13], [119], [188], [37], [0], [5], [40], [239], [135], [5], [15], [77], [203], [8], [13], [229], [240], [252], [41], [5], [66], [253], [213], [7], [0], [3], [16], [235], [166], [2], [0], [3], [82], [255], [245], [78], [0], [2], [19], [250], [225], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [254], [255], [238], [137], [68], [236], [255], [253], [254], [254], [247], [72], [69], [156], [93], [0], [200], [229], [0], [0], [0], [0], [0], [14], [142], [253], [244], [156], [20], [2], [1], [1], [0], [0], [0], [249], [3], [0], [16], [198], [225], [194], [3], [5], [78], [249], [102], [1], [11], [168], [1], [3], [207], [254], [224], [1], [1], [10], [180], [243], [34], [1], [0], [15], [228], [251], [239], [70], [5], [30], [205], [247], [214], [65], [8], [7], [205], [254], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [125], [242], [252], [241], [238], [235], [240], [156], [75], [82], [5], [7], [11], [4], [2], [10], [164], [32], [16], [10], [12], [8], [0], [0], [0], [44], [18], [18], [76], [209], [209], [192], [8], [8], [0], [32], [192], [77], [12], [8], [0], [77], [243], [56], [17], [6], [91], [115], [10], [19], [234], [17], [0], [0], [199], [231], [146], [15], [0], [15], [230], [232], [38], [14], [11], [152], [248], [255], [183], [1], [0], [24], [187], [253], [142], [66], [36], [182], [248], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [255], [255], [232], [211], [63], [33], [0], [91], [11], [65], [168], [88], [110], [35], [63], [15], [148], [231], [218], [189], [112], [132], [97], [37], [35], [38], [58], [5], [7], [0], [0], [24], [99], [88], [93], [36], [25], [0], [18], [132], [88], [32], [3], [108], [186], [191], [36], [0], [0], [110], [167], [230], [217], [34], [1], [0], [87], [200], [187], [35], [1], [34], [215], [225], [180], [62], [0], [165], [224], [251], [43], [2], [0], [101], [207], [74], [254], [231], [186], [181], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [255], [249], [203], [79], [14], [5], [56], [57], [79], [106], [239], [215], [199], [189], [187], [112], [78], [175], [85], [65], [79], [29], [14], [74], [191], [181], [198], [140], [76], [91], [56], [43], [2], [0], [0], [15], [170], [129], [48], [16], [0], [17], [165], [59], [12], [0], [19], [185], [46], [49], [22], [2], [71], [190], [213], [60], [21], [0], [0], [98], [205], [59], [15], [2], [74], [162], [188], [47], [147], [41], [176], [222], [66], [11], [0], [63], [202], [246], [136], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [254], [255], [255], [215], [200], [17], [49], [119], [91], [251], [250], [227], [179], [152], [54], [3], [0], [85], [75], [0], [0], [0], [5], [1], [0], [0], [0], [0], [0], [5], [48], [156], [178], [194], [194], [69], [68], [18], [0], [0], [3], [150], [71], [11], [0], [0], [143], [98], [24], [0], [1], [139], [154], [121], [71], [17], [4], [122], [203], [136], [23], [0], [0], [79], [174], [115], [19], [0], [7], [187], [224], [163], [0], [0], [132], [241], [108], [13], [0], [13], [168], [184], [255], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [252], [255], [255], [80], [174], [94], [190], [254], [196], [161], [150], [101], [0], [2], [88], [94], [91], [243], [195], [94], [95], [94], [104], [97], [88], [61], [48], [86], [84], [23], [0], [0], [1], [43], [126], [135], [145], [38], [64], [82], [0], [0], [43], [23], [69], [0], [0], [130], [86], [88], [0], [0], [1], [111], [178], [117], [79], [0], [55], [150], [111], [64], [0], [0], [3], [148], [125], [72], [0], [40], [183], [244], [84], [0], [0], [139], [244], [122], [48], [0], [1], [13], [176], [255], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [118], [250], [250], [190], [145], [53], [0], [49], [103], [113], [115], [235], [148], [136], [142], [144], [142], [142], [141], [142], [141], [162], [159], [89], [148], [160], [136], [111], [82], [39], [0], [1], [0], [0], [0], [96], [124], [75], [68], [0], [1], [104], [87], [18], [0], [51], [156], [105], [108], [65], [0], [42], [124], [224], [111], [14], [0], [93], [99], [96], [11], [0], [0], [127], [196], [109], [3], [35], [140], [235], [18], [0], [0], [126], [220], [185], [112], [5], [0], [49], [177], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [255], [253], [138], [215], [56], [18], [53], [132], [185], [242], [136], [123], [111], [6], [0], [37], [16], [0], [0], [0], [0], [0], [32], [73], [18], [20], [38], [111], [137], [182], [169], [123], [132], [109], [55], [2], [0], [1], [82], [75], [68], [32], [4], [95], [149], [22], [0], [26], [115], [138], [196], [72], [0], [1], [116], [146], [136], [89], [0], [2], [105], [103], [33], [0], [0], [95], [232], [116], [9], [2], [127], [136], [22], [0], [0], [85], [135], [232], [137], [112], [59], [78], [178], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [255], [191], [138], [29], [228], [152], [162], [191], [191], [106], [104], [10], [0], [0], [0], [0], [103], [99], [100], [139], [67], [73], [32], [10], [1], [0], [0], [0], [1], [13], [74], [102], [98], [107], [121], [116], [144], [114], [41], [0], [0], [54], [26], [3], [14], [245], [167], [26], [0], [0], [16], [100], [168], [117], [4], [0], [20], [96], [93], [123], [0], [0], [72], [100], [9], [0], [0], [91], [128], [128], [0], [6], [93], [104], [126], [2], [0], [10], [88], [191], [239], [174], [101], [82], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [245], [127], [52], [178], [255], [223], [86], [93], [50], [1], [13], [21], [115], [132], [140], [148], [120], [116], [79], [175], [141], [130], [143], [156], [137], [135], [138], [150], [128], [140], [18], [0], [0], [2], [19], [42], [82], [67], [149], [112], [38], [0], [0], [0], [8], [82], [108], [178], [147], [1], [0], [0], [64], [85], [130], [65], [0], [0], [13], [111], [127], [0], [0], [42], [116], [58], [1], [0], [22], [127], [77], [0], [0], [50], [135], [142], [13], [0], [0], [49], [231], [255], [208], [180], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [134], [117], [200], [245], [120], [55], [15], [3], [1], [115], [187], [189], [196], [70], [56], [60], [31], [26], [9], [38], [47], [36], [50], [60], [55], [54], [56], [60], [51], [138], [176], [167], [95], [10], [6], [0], [0], [1], [50], [78], [169], [150], [76], [1], [0], [0], [9], [78], [201], [169], [17], [0], [0], [7], [51], [132], [174], [6], [0], [16], [72], [137], [1], [0], [44], [75], [173], [10], [0], [23], [160], [10], [0], [0], [25], [150], [186], [23], [0], [0], [58], [143], [255], [250], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [224], [182], [164], [47], [14], [1], [125], [206], [206], [219], [76], [47], [39], [4], [0], [0], [10], [0], [52], [133], [169], [9], [12], [10], [11], [0], [0], [0], [0], [21], [44], [45], [84], [128], [183], [196], [85], [3], [0], [9], [41], [40], [49], [171], [123], [4], [0], [3], [38], [81], [196], [130], [43], [0], [0], [28], [50], [149], [73], [0], [5], [40], [167], [17], [0], [14], [120], [170], [15], [0], [34], [131], [75], [0], [0], [24], [56], [203], [161], [3], [0], [41], [104], [255], [255], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [255], [240], [79], [17], [0], [121], [219], [242], [250], [76], [28], [4], [0], [0], [0], [3], [138], [188], [201], [159], [131], [144], [157], [189], [190], [202], [178], [96], [84], [24], [2], [0], [0], [9], [19], [27], [43], [103], [195], [69], [0], [0], [0], [4], [25], [112], [200], [48], [0], [0], [5], [28], [181], [99], [52], [129], [1], [1], [22], [192], [92], [0], [1], [25], [143], [106], [0], [11], [25], [194], [6], [0], [19], [185], [140], [2], [0], [1], [49], [240], [187], [11], [8], [8], [160], [249], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [247], [30], [30], [78], [204], [244], [134], [40], [12], [2], [4], [35], [142], [191], [211], [213], [71], [10], [11], [8], [6], [6], [8], [10], [10], [11], [9], [5], [67], [8], [169], [114], [88], [57], [1], [0], [1], [5], [10], [86], [157], [6], [0], [0], [0], [5], [11], [192], [116], [0], [0], [0], [12], [158], [3], [9], [198], [32], [0], [10], [131], [80], [0], [0], [8], [198], [90], [0], [0], [99], [79], [0], [0], [9], [214], [225], [111], [0], [1], [92], [253], [244], [129], [18], [20], [94], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [12], [4], [177], [234], [141], [0], [0], [0], [27], [201], [240], [211], [168], [133], [0], [0], [1], [2], [6], [7], [6], [6], [26], [65], [29], [12], [5], [2], [1], [19], [0], [0], [1], [187], [193], [72], [4], [3], [0], [0], [0], [164], [188], [45], [6], [1], [0], [0], [177], [128], [5], [3], [0], [80], [165], [5], [1], [171], [83], [2], [0], [114], [203], [6], [0], [39], [216], [19], [1], [0], [42], [28], [1], [0], [0], [133], [248], [48], [5], [0], [80], [244], [253], [207], [40], [4], [250], [255], [255], [0], [0], [0], [0]], [[160], [105], [255], [29], [179], [214], [0], [0], [15], [21], [24], [206], [135], [40], [0], [2], [6], [15], [23], [48], [68], [197], [221], [217], [204], [196], [214], [225], [213], [160], [65], [65], [35], [15], [3], [0], [0], [0], [126], [128], [95], [11], [8], [0], [0], [33], [162], [199], [60], [13], [0], [0], [93], [181], [115], [6], [0], [117], [183], [17], [0], [84], [74], [4], [0], [62], [188], [22], [14], [27], [119], [45], [2], [0], [94], [38], [11], [0], [0], [91], [238], [192], [22], [0], [3], [181], [252], [255], [107], [210], [255], [255], [0], [0], [0], [0]], [[160], [105], [249], [230], [219], [10], [0], [24], [167], [212], [175], [2], [0], [14], [42], [64], [84], [127], [171], [176], [123], [39], [0], [13], [0], [0], [2], [3], [0], [26], [47], [122], [158], [125], [26], [7], [7], [12], [0], [0], [2], [96], [73], [34], [20], [5], [0], [19], [153], [116], [39], [13], [0], [0], [108], [64], [10], [0], [28], [140], [13], [0], [105], [57], [21], [0], [41], [188], [154], [5], [0], [0], [16], [6], [0], [84], [111], [23], [0], [0], [130], [236], [226], [30], [0], [0], [190], [252], [253], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [211], [85], [62], [64], [51], [162], [194], [160], [34], [14], [40], [124], [212], [193], [147], [22], [0], [0], [11], [16], [33], [22], [29], [54], [61], [47], [22], [6], [22], [0], [0], [9], [15], [29], [30], [69], [21], [37], [10], [0], [15], [142], [101], [74], [27], [0], [0], [47], [162], [86], [25], [0], [0], [57], [53], [11], [0], [5], [57], [27], [0], [97], [128], [19], [0], [64], [174], [54], [0], [0], [0], [54], [13], [0], [90], [113], [52], [0], [6], [113], [196], [194], [54], [6], [1], [203], [245], [255], [255], [255], [0], [0], [0], [0]], [[160], [105], [129], [3], [52], [243], [187], [151], [10], [0], [16], [115], [181], [182], [81], [1], [0], [4], [33], [64], [93], [99], [126], [62], [81], [151], [149], [132], [77], [54], [106], [49], [65], [64], [14], [0], [0], [50], [59], [104], [57], [54], [0], [0], [42], [143], [130], [54], [9], [0], [2], [79], [78], [58], [16], [0], [20], [34], [42], [0], [7], [87], [40], [0], [98], [87], [47], [0], [4], [91], [25], [14], [0], [79], [97], [25], [0], [35], [185], [50], [0], [0], [14], [205], [228], [89], [6], [58], [202], [244], [255], [255], [0], [0], [0], [0]], [[160], [105], [28], [51], [200], [201], [88], [0], [0], [39], [116], [206], [135], [16], [1], [30], [86], [82], [63], [123], [120], [103], [68], [2], [0], [0], [0], [0], [30], [71], [87], [92], [124], [122], [55], [66], [46], [0], [0], [0], [58], [108], [54], [0], [0], [0], [105], [163], [63], [23], [1], [0], [14], [112], [53], [38], [0], [9], [124], [56], [0], [24], [115], [19], [0], [67], [115], [50], [0], [0], [48], [74], [21], [0], [125], [61], [57], [3], [85], [168], [71], [0], [0], [60], [206], [234], [103], [0], [69], [168], [253], [255], [0], [0], [0], [0]], [[160], [105], [156], [179], [182], [70], [0], [12], [111], [158], [134], [84], [0], [80], [114], [101], [141], [101], [1], [0], [0], [0], [9], [24], [11], [69], [11], [64], [56], [25], [12], [0], [0], [0], [39], [91], [115], [95], [7], [0], [0], [8], [74], [52], [11], [0], [0], [82], [136], [31], [8], [50], [2], [0], [31], [101], [77], [1], [60], [131], [89], [0], [53], [92], [20], [0], [34], [83], [52], [0], [0], [66], [120], [40], [12], [18], [95], [97], [2], [99], [128], [88], [7], [0], [69], [196], [247], [92], [21], [49], [220], [251], [0], [0], [0], [0]], [[160], [105], [252], [173], [101], [6], [66], [148], [188], [104], [11], [0], [57], [169], [143], [60], [100], [59], [79], [74], [94], [49], [69], [103], [54], [109], [85], [88], [81], [51], [111], [97], [107], [57], [1], [0], [51], [96], [53], [69], [16], [0], [0], [52], [39], [59], [0], [0], [90], [2], [7], [56], [81], [3], [0], [49], [105], [93], [3], [54], [130], [100], [1], [66], [90], [0], [0], [15], [60], [80], [0], [0], [94], [145], [105], [0], [16], [125], [75], [0], [31], [92], [125], [20], [0], [64], [213], [229], [157], [24], [144], [251], [0], [0], [0], [0]], [[160], [105], [98], [133], [41], [65], [177], [106], [56], [0], [52], [66], [47], [65], [22], [11], [122], [158], [190], [66], [68], [36], [44], [58], [31], [30], [54], [18], [18], [20], [72], [71], [78], [74], [93], [55], [38], [0], [34], [50], [58], [51], [0], [0], [21], [75], [20], [0], [14], [52], [0], [5], [57], [21], [34], [0], [20], [85], [116], [22], [30], [146], [66], [0], [51], [18], [0], [0], [6], [90], [108], [0], [2], [77], [145], [101], [0], [23], [61], [55], [0], [4], [150], [155], [94], [0], [71], [149], [250], [171], [195], [255], [0], [0], [0], [0]], [[160], [105], [1], [49], [115], [98], [67], [3], [1], [82], [100], [34], [3], [0], [101], [84], [44], [60], [69], [6], [0], [0], [0], [0], [0], [0], [0], [0], [0], [4], [15], [0], [0], [17], [49], [71], [129], [75], [0], [0], [24], [49], [124], [4], [0], [17], [80], [2], [0], [33], [116], [6], [0], [10], [18], [109], [0], [10], [68], [156], [5], [38], [107], [45], [0], [9], [50], [5], [0], [17], [93], [143], [1], [0], [36], [207], [63], [0], [4], [80], [116], [0], [33], [111], [210], [151], [0], [26], [128], [255], [254], [255], [0], [0], [0], [0]], [[160], [105], [167], [100], [146], [27], [0], [29], [186], [150], [26], [2], [116], [66], [39], [28], [3], [0], [9], [120], [79], [57], [132], [151], [147], [159], [162], [118], [81], [102], [130], [171], [141], [0], [0], [15], [39], [46], [90], [0], [0], [8], [49], [102], [17], [0], [25], [41], [35], [5], [42], [68], [0], [0], [0], [44], [77], [0], [3], [52], [137], [2], [26], [39], [15], [0], [18], [131], [43], [0], [13], [182], [72], [0], [0], [59], [155], [3], [0], [18], [143], [104], [0], [13], [91], [240], [153], [0], [18], [202], [255], [255], [0], [0], [0], [0]], [[160], [105], [249], [67], [148], [111], [122], [218], [169], [28], [1], [81], [164], [15], [1], [16], [0], [134], [132], [47], [18], [13], [114], [201], [103], [42], [37], [27], [18], [23], [28], [46], [118], [153], [50], [0], [0], [4], [25], [129], [19], [0], [1], [24], [84], [17], [0], [9], [61], [85], [0], [15], [49], [101], [0], [1], [50], [118], [0], [0], [31], [95], [0], [5], [14], [48], [0], [29], [105], [72], [0], [30], [88], [69], [0], [1], [34], [127], [0], [0], [26], [199], [0], [0], [8], [125], [245], [181], [16], [32], [171], [255], [0], [0], [0], [0]], [[160], [105], [45], [5], [41], [243], [99], [44], [19], [27], [123], [52], [16], [0], [0], [64], [130], [31], [16], [43], [6], [20], [39], [20], [8], [1], [0], [19], [0], [0], [0], [1], [10], [25], [85], [198], [107], [0], [1], [15], [78], [102], [0], [0], [10], [132], [39], [0], [6], [51], [22], [0], [6], [12], [98], [1], [4], [44], [86], [0], [0], [11], [121], [0], [1], [27], [21], [0], [11], [173], [72], [0], [9], [182], [7], [0], [0], [140], [125], [0], [0], [126], [201], [66], [0], [10], [129], [250], [231], [44], [26], [249], [0], [0], [0], [0]], [[160], [105], [1], [0], [195], [155], [68], [116], [117], [226], [99], [1], [0], [183], [188], [38], [4], [1], [0], [138], [174], [7], [1], [0], [0], [39], [71], [144], [179], [160], [75], [0], [0], [0], [2], [13], [175], [132], [0], [0], [2], [9], [134], [0], [0], [4], [100], [15], [0], [1], [61], [8], [0], [0], [3], [57], [0], [1], [11], [157], [3], [0], [19], [88], [0], [1], [91], [25], [0], [16], [213], [0], [0], [16], [200], [37], [0], [4], [17], [86], [0], [42], [48], [220], [158], [0], [3], [78], [255], [250], [100], [250], [0], [0], [0], [0]], [[160], [105], [95], [180], [37], [0], [20], [212], [180], [27], [0], [5], [142], [133], [36], [0], [2], [131], [198], [104], [0], [0], [0], [33], [10], [0], [0], [0], [0], [0], [0], [18], [71], [2], [7], [7], [0], [123], [27], [7], [0], [0], [1], [134], [17], [1], [0], [46], [0], [7], [0], [36], [73], [8], [0], [0], [164], [8], [0], [1], [51], [28], [1], [18], [143], [2], [0], [100], [45], [0], [8], [203], [6], [0], [35], [211], [3], [1], [0], [55], [154], [9], [0], [78], [226], [26], [3], [0], [133], [254], [255], [243], [0], [0], [0], [0]], [[160], [105], [246], [92], [0], [5], [229], [163], [0], [0], [12], [103], [72], [0], [11], [27], [56], [195], [29], [0], [0], [0], [0], [1], [0], [0], [2], [0], [2], [1], [4], [1], [0], [35], [136], [146], [13], [0], [147], [130], [15], [2], [0], [0], [170], [42], [1], [0], [0], [125], [6], [0], [9], [152], [20], [0], [11], [149], [21], [0], [0], [141], [45], [0], [64], [34], [1], [0], [133], [4], [0], [166], [129], [5], [0], [70], [87], [49], [4], [0], [46], [179], [24], [0], [7], [194], [88], [26], [10], [197], [239], [255], [0], [0], [0], [0]], [[160], [105], [154], [79], [41], [37], [137], [0], [2], [50], [88], [17], [0], [0], [107], [211], [152], [14], [30], [45], [37], [4], [4], [7], [0], [0], [12], [3], [15], [6], [30], [4], [1], [0], [1], [135], [94], [34], [0], [10], [103], [24], [12], [0], [0], [127], [7], [6], [0], [3], [44], [16], [0], [8], [132], [28], [0], [37], [142], [29], [0], [6], [130], [17], [0], [9], [8], [0], [0], [30], [0], [0], [178], [72], [12], [0], [163], [214], [69], [6], [0], [29], [193], [38], [0], [26], [220], [226], [112], [1], [99], [253], [0], [0], [0], [0]], [[160], [105], [33], [35], [205], [113], [53], [58], [70], [182], [23], [23], [16], [43], [136], [107], [2], [42], [146], [177], [145], [45], [7], [4], [0], [0], [23], [35], [27], [35], [21], [40], [6], [9], [0], [0], [24], [129], [19], [32], [0], [30], [46], [37], [0], [0], [0], [37], [0], [0], [0], [61], [31], [0], [0], [118], [21], [0], [3], [115], [44], [0], [2], [68], [16], [0], [1], [0], [13], [7], [50], [0], [25], [153], [103], [0], [1], [140], [224], [76], [7], [0], [113], [196], [54], [0], [121], [240], [230], [58], [0], [240], [0], [0], [0], [0]], [[160], [105], [153], [123], [124], [1], [48], [227], [154], [9], [0], [78], [110], [110], [11], [0], [49], [129], [84], [19], [8], [90], [36], [51], [53], [36], [64], [88], [66], [86], [52], [105], [9], [71], [67], [45], [0], [0], [44], [80], [21], [0], [6], [91], [47], [0], [0], [40], [16], [36], [0], [3], [77], [43], [0], [28], [76], [1], [0], [9], [126], [22], [0], [5], [46], [23], [0], [0], [32], [19], [192], [31], [0], [5], [199], [74], [68], [141], [145], [206], [89], [0], [1], [131], [195], [74], [15], [147], [170], [226], [78], [205], [0], [0], [0], [0]], [[160], [105], [230], [139], [0], [24], [172], [143], [31], [2], [73], [132], [126], [8], [0], [71], [83], [42], [1], [0], [76], [115], [122], [102], [92], [62], [14], [0], [0], [0], [0], [11], [4], [91], [116], [106], [69], [60], [0], [3], [36], [61], [0], [0], [81], [21], [0], [0], [28], [87], [40], [0], [1], [80], [7], [0], [41], [67], [0], [0], [30], [75], [10], [0], [11], [39], [56], [0], [0], [4], [133], [136], [0], [0], [108], [226], [222], [159], [1], [140], [204], [66], [0], [7], [149], [222], [110], [0], [10], [206], [239], [203], [0], [0], [0], [0]], [[160], [105], [131], [89], [8], [92], [114], [0], [37], [118], [193], [126], [5], [58], [111], [99], [9], [9], [97], [85], [132], [113], [92], [16], [0], [0], [23], [84], [78], [78], [31], [61], [20], [6], [0], [35], [86], [152], [57], [0], [0], [76], [76], [3], [0], [26], [42], [0], [0], [30], [53], [61], [0], [7], [12], [8], [0], [83], [61], [0], [0], [46], [63], [8], [0], [0], [76], [86], [23], [0], [16], [170], [92], [0], [4], [170], [241], [117], [0], [22], [152], [198], [67], [0], [55], [152], [214], [103], [0], [71], [160], [244], [0], [0], [0], [0]], [[160], [105], [3], [5], [109], [96], [34], [16], [122], [206], [93], [7], [118], [169], [175], [10], [0], [88], [153], [78], [34], [2], [0], [13], [78], [104], [116], [99], [72], [72], [29], [56], [19], [77], [60], [0], [0], [70], [57], [53], [0], [0], [70], [23], [7], [0], [38], [41], [0], [0], [2], [56], [55], [0], [3], [88], [1], [1], [56], [49], [0], [0], [46], [49], [9], [0], [5], [79], [101], [5], [0], [61], [158], [48], [0], [45], [113], [194], [25], [0], [45], [141], [186], [83], [0], [33], [102], [221], [131], [0], [28], [118], [0], [0], [0], [0]], [[160], [105], [1], [158], [170], [26], [37], [75], [161], [103], [0], [105], [207], [77], [49], [0], [0], [119], [43], [0], [0], [0], [72], [68], [180], [69], [63], [15], [1], [0], [0], [0], [1], [51], [40], [96], [0], [0], [3], [35], [35], [22], [0], [13], [68], [46], [0], [27], [0], [0], [0], [0], [54], [0], [0], [54], [41], [0], [0], [37], [23], [0], [0], [28], [71], [0], [0], [0], [128], [97], [0], [0], [49], [162], [6], [0], [8], [122], [146], [0], [0], [24], [91], [200], [30], [0], [5], [84], [223], [116], [16], [24], [0], [0], [0], [0]], [[160], [105], [166], [221], [48], [50], [128], [116], [40], [17], [1], [88], [61], [0], [0], [40], [53], [31], [0], [1], [101], [59], [34], [43], [109], [0], [0], [2], [18], [0], [27], [113], [61], [0], [0], [45], [28], [6], [0], [0], [17], [33], [2], [0], [30], [106], [23], [0], [0], [75], [0], [0], [9], [74], [0], [0], [19], [25], [0], [2], [18], [16], [0], [0], [30], [136], [5], [0], [36], [126], [1], [0], [0], [70], [148], [0], [0], [18], [86], [145], [8], [0], [4], [147], [184], [45], [0], [0], [74], [231], [187], [186], [0], [0], [0], [0]], [[160], [105], [233], [63], [12], [67], [210], [136], [76], [52], [57], [13], [0], [0], [95], [149], [17], [0], [17], [129], [32], [19], [5], [126], [15], [0], [10], [126], [12], [0], [9], [36], [42], [56], [2], [0], [9], [2], [21], [0], [0], [7], [102], [36], [0], [27], [55], [14], [0], [24], [87], [0], [0], [27], [61], [0], [0], [8], [33], [0], [2], [124], [24], [0], [0], [60], [137], [0], [0], [26], [81], [0], [0], [3], [136], [153], [1], [0], [8], [46], [150], [2], [0], [25], [195], [197], [64], [0], [5], [187], [255], [255], [0], [0], [0], [0]], [[160], [105], [98], [3], [19], [196], [194], [21], [84], [115], [32], [62], [14], [110], [155], [143], [0], [8], [171], [112], [0], [0], [96], [190], [0], [103], [142], [24], [1], [0], [0], [0], [4], [11], [30], [68], [0], [0], [12], [26], [0], [0], [20], [67], [10], [0], [9], [64], [0], [0], [78], [42], [0], [1], [12], [48], [0], [0], [13], [59], [0], [23], [78], [0], [0], [3], [151], [26], [0], [0], [16], [155], [3], [0], [18], [155], [155], [1], [0], [0], [170], [144], [25], [0], [27], [182], [202], [119], [0], [29], [229], [253], [0], [0], [0], [0]], [[160], [105], [7], [0], [182], [122], [15], [2], [106], [176], [2], [6], [93], [10], [13], [31], [0], [118], [119], [8], [0], [116], [118], [89], [153], [92], [13], [0], [12], [8], [37], [12], [3], [0], [3], [6], [0], [0], [1], [2], [81], [8], [0], [6], [85], [6], [0], [6], [78], [0], [65], [123], [6], [0], [0], [12], [15], [0], [1], [170], [22], [0], [7], [108], [0], [0], [11], [159], [101], [0], [0], [39], [171], [7], [0], [12], [209], [127], [0], [0], [28], [190], [212], [30], [0], [13], [225], [226], [120], [0], [18], [253], [0], [0], [0], [0]], [[160], [105], [1], [203], [240], [64], [0], [80], [248], [144], [17], [69], [79], [4], [0], [29], [0], [109], [1], [0], [0], [115], [1], [2], [148], [0], [0], [53], [132], [0], [2], [174], [135], [86], [0], [0], [0], [43], [4], [0], [0], [141], [10], [0], [0], [66], [9], [0], [0], [52], [4], [1], [65], [0], [0], [0], [151], [3], [0], [2], [184], [1], [0], [1], [118], [0], [0], [1], [172], [33], [0], [0], [154], [114], [0], [0], [57], [193], [38], [0], [0], [2], [198], [232], [1], [0], [52], [11], [210], [93], [24], [176], [0], [0], [0], [0]], [[160], [105], [14], [237], [201], [3], [61], [208], [167], [0], [79], [138], [0], [0], [112], [0], [0], [4], [0], [0], [114], [0], [0], [53], [0], [0], [6], [122], [1], [0], [0], [0], [0], [16], [62], [21], [4], [0], [76], [7], [0], [0], [89], [7], [0], [0], [34], [6], [0], [37], [34], [0], [0], [38], [6], [0], [0], [108], [3], [0], [0], [17], [1], [0], [52], [65], [0], [0], [40], [174], [24], [0], [0], [142], [19], [0], [0], [0], [130], [58], [1], [0], [45], [203], [185], [7], [0], [0], [194], [200], [44], [139], [0], [0], [0], [0]], [[160], [105], [201], [245], [81], [14], [52], [154], [2], [0], [103], [107], [0], [7], [69], [0], [10], [3], [0], [10], [110], [0], [0], [53], [0], [0], [86], [0], [0], [5], [20], [9], [5], [20], [0], [87], [53], [0], [0], [61], [9], [0], [1], [73], [15], [0], [0], [77], [0], [0], [62], [11], [0], [0], [72], [4], [0], [83], [47], [0], [0], [56], [12], [0], [1], [136], [32], [0], [0], [30], [144], [0], [0], [0], [175], [20], [0], [0], [3], [152], [44], [0], [0], [111], [232], [123], [13], [0], [78], [214], [237], [227], [0], [0], [0], [0]], [[160], [105], [197], [222], [8], [79], [207], [114], [0], [0], [219], [44], [0], [24], [10], [0], [84], [16], [0], [63], [11], [3], [19], [0], [0], [21], [83], [0], [7], [49], [113], [51], [28], [0], [7], [0], [12], [20], [3], [0], [46], [0], [0], [0], [82], [2], [0], [12], [11], [0], [0], [67], [7], [0], [0], [31], [0], [0], [78], [19], [0], [0], [31], [10], [0], [3], [185], [15], [0], [10], [153], [37], [0], [0], [122], [154], [34], [0], [0], [127], [185], [36], [0], [0], [154], [196], [119], [25], [0], [116], [249], [238], [0], [0], [0], [0]], [[160], [105], [44], [168], [51], [11], [189], [51], [0], [25], [132], [0], [0], [91], [2], [19], [149], [5], [11], [163], [0], [9], [83], [0], [3], [117], [0], [0], [42], [76], [0], [0], [0], [0], [24], [4], [0], [58], [29], [0], [0], [0], [0], [0], [0], [22], [1], [0], [36], [9], [0], [14], [43], [0], [0], [35], [8], [0], [0], [89], [3], [0], [1], [74], [6], [0], [112], [80], [0], [0], [81], [185], [7], [0], [1], [149], [168], [3], [0], [0], [141], [183], [47], [0], [90], [115], [210], [154], [8], [26], [207], [230], [0], [0], [0], [0]], [[160], [105], [29], [165], [61], [25], [180], [35], [0], [89], [79], [3], [3], [107], [0], [61], [108], [10], [72], [109], [0], [0], [71], [0], [15], [157], [0], [0], [66], [0], [0], [4], [22], [0], [0], [9], [3], [0], [47], [12], [0], [0], [9], [3], [0], [33], [30], [0], [0], [22], [1], [0], [42], [0], [0], [0], [20], [1], [0], [60], [63], [0], [0], [116], [68], [0], [0], [99], [23], [0], [0], [147], [67], [0], [0], [21], [134], [59], [0], [0], [51], [147], [187], [64], [0], [22], [131], [231], [90], [58], [125], [244], [0], [0], [0], [0]], [[160], [105], [20], [186], [0], [1], [169], [2], [0], [100], [106], [4], [44], [117], [0], [80], [2], [0], [153], [0], [0], [41], [31], [0], [14], [79], [0], [0], [87], [0], [0], [55], [47], [6], [0], [1], [19], [2], [0], [33], [18], [0], [14], [11], [0], [0], [48], [14], [0], [2], [42], [0], [0], [3], [3], [0], [3], [29], [0], [1], [106], [12], [0], [40], [153], [0], [0], [45], [85], [0], [0], [21], [133], [1], [0], [0], [14], [160], [17], [0], [0], [21], [216], [197], [31], [2], [10], [119], [201], [112], [156], [252], [0], [0], [0], [0]], [[160], [105], [18], [125], [0], [65], [147], [0], [0], [124], [36], [0], [118], [48], [0], [138], [0], [0], [104], [0], [0], [99], [10], [0], [10], [0], [0], [0], [101], [0], [5], [57], [14], [7], [4], [0], [16], [48], [0], [17], [53], [0], [0], [8], [12], [0], [6], [63], [0], [0], [63], [7], [0], [4], [25], [0], [0], [56], [0], [0], [18], [83], [0], [0], [76], [42], [0], [0], [63], [38], [0], [0], [90], [82], [0], [0], [0], [96], [114], [18], [0], [6], [100], [221], [134], [0], [5], [9], [194], [228], [208], [247], [0], [0], [0], [0]], [[160], [105], [147], [61], [5], [145], [152], [0], [17], [170], [11], [0], [112], [0], [0], [161], [0], [0], [92], [0], [0], [76], [6], [0], [21], [0], [0], [0], [74], [0], [7], [93], [0], [0], [22], [1], [0], [48], [10], [0], [27], [27], [0], [0], [53], [0], [0], [42], [0], [0], [14], [43], [0], [0], [29], [0], [0], [20], [13], [0], [0], [90], [9], [0], [0], [78], [0], [0], [5], [71], [54], [0], [27], [124], [20], [0], [0], [6], [176], [127], [12], [0], [1], [159], [208], [94], [38], [18], [150], [232], [215], [242], [0], [0], [0], [0]], [[160], [105], [229], [19], [3], [193], [161], [5], [83], [183], [12], [0], [127], [0], [0], [67], [0], [0], [172], [0], [0], [52], [68], [0], [22], [0], [0], [12], [68], [0], [2], [157], [1], [0], [12], [77], [0], [6], [76], [0], [0], [54], [0], [0], [26], [3], [0], [2], [8], [0], [0], [43], [69], [0], [6], [79], [0], [0], [107], [0], [0], [28], [69], [0], [0], [27], [18], [0], [0], [38], [128], [0], [0], [34], [78], [77], [0], [0], [59], [168], [134], [0], [0], [53], [178], [122], [166], [8], [92], [236], [226], [209], [0], [0], [0], [0]], [[160], [105], [217], [1], [59], [202], [46], [3], [156], [113], [3], [48], [177], [0], [0], [60], [0], [0], [176], [4], [0], [88], [43], [0], [6], [0], [0], [60], [64], [0], [0], [43], [29], [0], [0], [110], [0], [0], [30], [33], [0], [16], [52], [0], [0], [23], [0], [0], [4], [2], [0], [9], [96], [0], [0], [123], [0], [0], [69], [112], [0], [4], [123], [1], [0], [1], [77], [2], [0], [6], [121], [0], [0], [0], [45], [153], [4], [0], [0], [42], [185], [145], [0], [6], [44], [140], [225], [91], [44], [209], [233], [229], [0], [0], [0], [0]], [[160], [105], [221], [0], [99], [184], [5], [8], [191], [107], [11], [140], [113], [0], [0], [128], [10], [0], [65], [118], [0], [96], [4], [0], [1], [2], [0], [15], [31], [0], [6], [0], [49], [0], [0], [22], [26], [0], [0], [64], [0], [0], [18], [0], [0], [6], [8], [0], [0], [34], [1], [12], [21], [0], [0], [72], [1], [0], [8], [118], [4], [0], [38], [36], [0], [0], [64], [88], [0], [0], [45], [143], [0], [0], [5], [155], [88], [6], [0], [0], [37], [195], [152], [1], [0], [102], [216], [211], [38], [185], [222], [233], [0], [0], [0], [0]], [[160], [105], [206], [3], [183], [126], [1], [11], [200], [64], [2], [21], [71], [0], [0], [170], [2], [0], [25], [98], [0], [28], [27], [0], [0], [120], [0], [0], [45], [0], [16], [3], [7], [14], [18], [0], [45], [4], [0], [18], [66], [0], [1], [0], [0], [0], [6], [0], [0], [92], [111], [70], [76], [48], [0], [8], [21], [0], [0], [42], [27], [0], [2], [29], [0], [0], [7], [114], [3], [0], [7], [182], [18], [0], [0], [20], [44], [132], [11], [0], [0], [102], [199], [95], [0], [12], [211], [219], [190], [227], [215], [252], [0], [0], [0], [0]], [[160], [105], [151], [42], [210], [196], [61], [9], [162], [39], [0], [17], [153], [0], [0], [34], [0], [0], [33], [5], [0], [1], [154], [0], [0], [86], [0], [0], [102], [0], [5], [33], [0], [43], [106], [0], [5], [60], [0], [1], [43], [0], [0], [8], [0], [0], [12], [0], [0], [6], [59], [46], [5], [90], [0], [0], [63], [0], [0], [2], [146], [0], [0], [2], [76], [0], [0], [41], [74], [0], [35], [178], [124], [0], [0], [0], [2], [94], [80], [28], [0], [9], [187], [186], [5], [0], [25], [156], [216], [243], [247], [242], [0], [0], [0], [0]], [[160], [105], [211], [212], [227], [160], [5], [0], [117], [104], [0], [11], [193], [10], [0], [126], [59], [0], [81], [23], [0], [0], [189], [0], [0], [1], [1], [0], [77], [1], [0], [43], [12], [0], [100], [20], [0], [13], [1], [1], [0], [62], [0], [0], [31], [0], [0], [0], [0], [0], [83], [0], [0], [71], [1], [0], [5], [46], [0], [0], [47], [14], [0], [0], [117], [145], [0], [0], [65], [121], [0], [44], [159], [4], [69], [157], [3], [0], [31], [174], [62], [0], [14], [118], [121], [1], [0], [9], [204], [217], [251], [239], [0], [0], [0], [0]], [[160], [105], [240], [151], [217], [38], [1], [8], [174], [29], [0], [0], [182], [2], [0], [168], [42], [7], [18], [97], [0], [0], [122], [1], [0], [83], [81], [0], [2], [54], [1], [9], [133], [0], [13], [96], [0], [0], [67], [0], [0], [45], [5], [0], [5], [14], [0], [13], [0], [0], [17], [4], [0], [2], [20], [0], [0], [11], [2], [0], [0], [57], [1], [0], [3], [104], [5], [0], [1], [154], [5], [40], [76], [75], [0], [71], [43], [0], [0], [105], [192], [13], [0], [31], [168], [66], [3], [0], [150], [235], [243], [229], [0], [0], [0], [0]], [[160], [105], [245], [146], [196], [50], [35], [78], [153], [34], [0], [1], [118], [20], [0], [40], [65], [0], [1], [114], [0], [0], [38], [18], [0], [112], [95], [0], [0], [142], [9], [0], [125], [8], [0], [77], [8], [0], [26], [6], [0], [2], [57], [0], [0], [7], [0], [0], [0], [0], [2], [53], [0], [0], [99], [3], [0], [0], [29], [0], [0], [70], [29], [0], [0], [43], [46], [1], [0], [54], [39], [187], [182], [148], [30], [11], [59], [25], [0], [4], [128], [132], [23], [0], [87], [166], [60], [22], [172], [220], [241], [229], [0], [0], [0], [0]], [[160], [105], [255], [208], [202], [185], [172], [19], [125], [56], [0], [9], [99], [43], [1], [9], [122], [0], [0], [120], [6], [0], [27], [36], [0], [5], [126], [0], [0], [114], [0], [0], [90], [49], [0], [5], [67], [2], [0], [28], [5], [0], [44], [3], [0], [13], [13], [0], [0], [0], [0], [60], [11], [0], [38], [17], [0], [0], [26], [0], [0], [1], [106], [14], [4], [0], [48], [19], [0], [1], [91], [191], [131], [159], [164], [29], [23], [129], [13], [0], [7], [159], [152], [45], [18], [125], [192], [141], [224], [233], [246], [247], [0], [0], [0], [0]], [[160], [105], [245], [212], [233], [233], [213], [87], [126], [136], [0], [10], [184], [67], [1], [43], [154], [0], [0], [146], [41], [0], [17], [72], [0], [0], [129], [5], [0], [1], [23], [0], [3], [44], [0], [0], [120], [6], [0], [1], [34], [0], [3], [13], [0], [0], [43], [0], [0], [0], [0], [20], [51], [0], [0], [27], [0], [0], [4], [8], [0], [0], [86], [73], [10], [0], [3], [75], [3], [0], [13], [148], [105], [165], [172], [4], [0], [69], [83], [41], [0], [61], [136], [185], [92], [6], [161], [189], [226], [219], [247], [225], [0], [0], [0], [0]], [[160], [105], [206], [187], [234], [217], [167], [65], [139], [118], [12], [0], [172], [38], [0], [45], [93], [3], [0], [131], [109], [8], [7], [67], [0], [0], [105], [58], [0], [0], [84], [1], [0], [23], [31], [0], [57], [3], [0], [0], [53], [0], [0], [19], [0], [0], [31], [0], [0], [4], [0], [0], [55], [3], [0], [69], [3], [0], [4], [17], [24], [0], [20], [97], [32], [0], [0], [123], [60], [0], [11], [70], [109], [110], [114], [64], [5], [14], [112], [153], [33], [0], [21], [153], [198], [68], [130], [212], [226], [231], [244], [220], [0], [0], [0], [0]], [[160], [105], [187], [198], [220], [202], [129], [73], [146], [208], [53], [1], [168], [104], [0], [51], [98], [47], [0], [63], [155], [11], [0], [19], [9], [0], [59], [99], [0], [0], [59], [10], [0], [2], [71], [0], [1], [6], [17], [0], [18], [6], [0], [11], [14], [0], [32], [0], [0], [9], [0], [0], [34], [12], [0], [56], [32], [0], [0], [4], [101], [8], [0], [48], [101], [2], [0], [71], [136], [0], [0], [11], [45], [3], [1], [145], [65], [1], [27], [135], [144], [0], [2], [89], [203], [198], [202], [211], [233], [242], [250], [233], [0], [0], [0], [0]], [[160], [105], [213], [238], [217], [196], [207], [126], [184], [212], [156], [92], [140], [151], [2], [51], [133], [107], [0], [22], [192], [0], [0], [30], [86], [0], [0], [92], [49], [0], [5], [75], [0], [0], [28], [11], [0], [6], [35], [0], [0], [39], [0], [0], [37], [0], [0], [13], [0], [4], [40], [0], [7], [10], [0], [13], [80], [21], [0], [3], [74], [41], [0], [0], [101], [30], [0], [0], [132], [31], [0], [0], [60], [63], [2], [67], [130], [76], [1], [36], [206], [88], [5], [19], [174], [220], [205], [218], [251], [237], [239], [252], [0], [0], [0], [0]], [[160], [105], [230], [230], [197], [195], [231], [144], [165], [220], [226], [126], [105], [211], [111], [2], [132], [95], [0], [0], [182], [0], [0], [31], [117], [0], [0], [46], [128], [0], [0], [97], [40], [0], [0], [73], [0], [0], [14], [38], [0], [64], [0], [0], [19], [4], [0], [14], [30], [0], [34], [0], [0], [69], [10], [0], [119], [26], [10], [0], [3], [40], [34], [0], [34], [93], [2], [0], [72], [83], [16], [0], [44], [142], [15], [0], [63], [153], [10], [0], [129], [202], [82], [0], [68], [193], [203], [242], [228], [247], [238], [248], [0], [0], [0], [0]], [[160], [105], [239], [234], [216], [226], [239], [185], [212], [226], [229], [81], [110], [224], [91], [1], [103], [114], [3], [0], [158], [99], [0], [22], [115], [0], [0], [0], [164], [0], [0], [25], [112], [0], [0], [92], [88], [0], [1], [47], [28], [64], [0], [0], [1], [43], [0], [2], [71], [3], [2], [17], [0], [41], [26], [0], [134], [55], [5], [22], [0], [8], [100], [0], [1], [52], [47], [0], [9], [136], [104], [0], [13], [72], [138], [38], [7], [178], [82], [0], [27], [205], [180], [11], [3], [96], [207], [240], [229], [249], [246], [251], [0], [0], [0], [0]], [[41], [27], [62], [62], [59], [60], [62], [49], [60], [59], [59], [19], [31], [55], [12], [0], [25], [33], [1], [0], [39], [38], [0], [4], [32], [0], [0], [0], [43], [0], [0], [0], [34], [0], [0], [21], [33], [0], [0], [10], [11], [16], [0], [0], [0], [16], [0], [0], [21], [1], [0], [6], [0], [2], [8], [0], [34], [19], [0], [8], [0], [0], [31], [0], [0], [5], [17], [0], [0], [40], [36], [0], [0], [8], [50], [15], [1], [48], [29], [0], [0], [50], [52], [4], [0], [15], [53], [61], [61], [63], [63], [65], [0], [0], [0], [0]], [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]]]
      }
    ]
  }


# In[47]:


with open('request-fingerprint.json', 'w') as convert_file:
     convert_file.write(json.dumps(req))


# In[ ]:




