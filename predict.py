#!/usr/bin/env python
# coding: utf-8

# 
# Read an image file and make a fingerprint prediction.
#

import numpy
import tensorflow as tf
import os
import cv2
import random
from PIL import Image
from tensorflow.keras.models import load_model

def single_prediction(img_array: numpy.array)-> list:
    """
    Args:
         img_array - The image as a numpy array
    Returns:
         A float value that represents the confidence of the prediction.
    """
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

if __name__ == '__main__':

    fingerprint_model = load_model('./models/model-20-epochs.h5')

    #
    # Make a few predictions from a single image files.
    #
    img_size = 96
    datapath = './data/fingerprint_real/'
    F_Left_index = cv2.imread(f'{datapath}103__F_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(F_Left_index, (img_size, img_size))
    img_resize.resize(1, 96, 96, 1)
    print(f'Prediction = {single_prediction(img_resize)}')

    F_Left_thumb = cv2.imread(f'{datapath}275__F_Left_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(F_Left_thumb, (img_size, img_size))
    img_resize.resize(1, 96, 96, 1)
    print(f'Prediction = {single_prediction(img_resize)}')

    M_Right_index = cv2.imread(f'{datapath}232__M_Right_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(M_Right_index, (img_size, img_size))
    img_resize.resize(1, 96, 96, 1)
    print(f'Prediction = {single_prediction(img_resize)}')

    M_Right_index = cv2.imread(f'{datapath}504__M_Right_index_finger.BMP', cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(M_Right_index, (img_size, img_size))
    img_resize.resize(1, 96, 96, 1)
    print(f'Prediction = {single_prediction(img_resize)}')