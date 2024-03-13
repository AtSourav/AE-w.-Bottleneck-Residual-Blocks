"""
@author: sourav

We define a model to calculated feature losses for an AE/VAE. 
We use the image classification model ResNet50V2 to compute the feature maps.
ResNet50V2 allows an input tensor shape of width 32 or greater,
so it's suitable to use on cifar10. 
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

from keras.applications.resnet_v2 import ResNet50V2



class feature_model():
    '''
    Instantiates a ResNet50V2 model with weights pre-trained on ImageNet. 
    The input is an image as a keras tensor which could either be the output
    of the (V)AE or an original image from the dataset. 

    Arguments:
    -- map_layers: a list of (lower level) layers (specified by integers) that would be used to extract maps.
    -- inp_shape: the shape of the input image, will depend on the dataset we're working on.
                  It should have 3 channels and the height and the width should be no less than 32.
                  For example, (64,64,3) is a valid input shape.
    -- inp_tensor: the input images to be fed into the feature model in batches.

    Attributes:
    -- feature_maps: a list of feature maps to be extracted from the layers specified.
    -- f_model: a keras Model object that outputs the feature maps.
    '''


    def __init__(self, map_layers, inp_tensor):

        inp_shape = K.int_shape(inp_tensor)[1:]     # the inp_tensor will be of the shape (batch, H, W, C)

        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=inp_shape)
        base_model.trainable = False

        self.feature_maps = [base_model.layers[i].output for i in map_layers]

        self.f_model = keras.Model(inp_tensor, self.feature_maps, name='feature_model')







