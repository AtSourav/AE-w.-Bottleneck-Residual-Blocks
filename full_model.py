"""
@author: sourav

We define the full model to calculate the perceptual losses by combining the feature_model and the main_model. 
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import initializers
from keras import layers

from feature_model import *
from main_model import *
from Residual_blocks import *


class AE_perceptual():
    '''
    Has an attribute that is an autoencoder combined with a perceptual model (ResNet50V2) to calculate
    perceptual losses. The outputs are both the reconstructed image from the autoencoder
    and the feature maps obtained from applying the feature model on the reconstructions.
    -- Add the architecture.
    '''

    def __init__(self, inp_shape, map_layers, initializer, latent_dim, *args, **kwargs):
        super().__init__()

        full_model_input = keras.Input(shape=inp_shape)

        self.AE = AE(initializer, latent_dim)

        feature_model_input = self.AE(full_model_input)

        self.feature_model = feature_model(map_layers, feature_model_input).f_model

        feature_maps = self.feature_model(feature_model_input)
        full_model_output = [feature_model_input] + feature_maps

        self.full_model = keras.Model(full_model_input, full_model_output, name='full_model')

