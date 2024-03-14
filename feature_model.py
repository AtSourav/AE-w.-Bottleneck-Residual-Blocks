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

    Arguments:
    -- inp_shape: the shape of the input image, will depend on the dataset we're working on.
                  It should have 3 channels and the height and the width should be no less than 32.
                  For example, (64,64,3) is a valid input shape.
    -- num_feature_maps: the number of feature maps we wish to extract.

    Attributes:
    -- base_model: an instance of ResNet50V2 pre-trained on ImageNet.
    -- conv_layers: a list of names of all the convolutional layers in the base_model.
    -- feat_layers: a list of names of convolutional layers where the feature maps are extracted from.
    -- feature_maps: a list of feature maps to be extracted from the layers specified (symbolic Keras tensors).
    -- f_model: a keras Model object that outputs the feature maps.
    -- output: the output of f_model

    Methods:
    -- summarise(show_trainable): provides a model summary of f_model. The show-trainable option is True by default.
                                  It indicates if the layers are trainable or not.
    '''


    def __init__(self, inp_shape, num_feature_maps):

        self.base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=inp_shape)
        self.base_model.trainable = False

        self.conv_layers = [layer.name for layer in self.base_model.layers if layer.name[-4:]=='conv']
        m = int(len(self.conv_layers)/2)                 # takes the floor integer
        s = int(m/num_feature_maps)
        self.feat_layers = self.conv_layers[s-1:m:s]      # we are taking num_feature_maps number of feature layers
                                                     # evenly spaced in the first half of the list of all conv layers
        
        self.feature_maps = [self.base_model.get_layer(name).output for name in self.feat_layers]

        self.f_model = keras.Model(inputs=self.base_model.input, outputs=self.feature_maps, name='feature_model')

        self.output = self.f_model.output

    def summarise(self, show_trainable=True):
        self.f_model.summary(show_trainable=show_trainable)







