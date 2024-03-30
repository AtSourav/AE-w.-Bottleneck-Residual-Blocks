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


class AE_perceptual():
    '''
    Has an attribute that is an autoencoder combined with a perceptual model (ResNet50V2) to calculate
    perceptual losses. The outputs are both the reconstructed image from the autoencoder
    and the feature maps obtained from applying the feature model on the reconstructions.
    -- Add the architecture.

    Attributes:
    -- AE: an instance of the Autoencoder based on residual blocks defined in the main_model module.
    -- feature_model: an instance of the feature_model class that provides the pre-trained ResNet50V2 
                        model used for calculating the feature losses as an attribute. Refer to the 
                        feature_model module.
    -- full_model: a composition of the AE and the pre-trained ResNet50V2 to calculate feature maps.
    -- output: output of the full_model. It is a list of feature maps and the reconstructed image (last).

    Methods:
    --summarise(show_trainable): provides a model summary of the full model with the number of 
                                    trainable and non-trainable parameters.  
    '''

    def __init__(self, inp_shape, num_feature_maps, initializer, latent_dim, *args, **kwargs):

        inputs = keras.Input(shape=inp_shape)

        self.AE = AE(initializer, latent_dim, inp_shape)

        AE_out = self.AE(inputs)

        self.feature_model = feature_model(inp_shape, num_feature_maps).f_model

        feature_maps = self.feature_model(AE_out)
        reconstruction = self.AE.output
        feature_maps.append(reconstruction)   # appending the reconstruction to the feature maps list
                                                                    # produced by the feature model
        

        self.full_model = keras.Model(self.AE.input, feature_maps, name='full_model')
        self.output = self.full_model.output

    def summarise(self):
        self.full_model.summary()






class VAE_perceptual():
    '''
    Has an attribute that is a variational autoencoder combined with a perceptual model 
    (ResNet50V2) to calculate perceptual losses. The outputs are both the reconstructed 
    image from the autoencoder and the feature maps obtained from applying the feature 
    model on the reconstructions.
    -- Add the architecture.

    Attributes:
    -- VAE: an instance of the VAE based on residual blocks defined in the main_model 
            module.
    -- feature_model: an instance of the feature_model class that provides the pre-trained 
                        ResNet50V2 model used for calculating the feature losses as an 
                        attribute. Refer to the feature_model module.
    -- full_model: a composition of the VAE and the pre-trained ResNet50V2 to calculate 
                    feature maps.
    -- output: output of the full_model. It is a list of feature maps and the generated 
                image (last).

    Methods:
    --summarise(): provides a model summary of the full model with the number of trainable 
                    and non-trainable parameters.  
    '''

    def __init__(self, inp_shape, num_feature_maps, initializer, latent_dim, seed, *args, **kwargs):

        inputs = keras.Input(shape=inp_shape)

        self.VAE = VAE(initializer, latent_dim, seed, inp_shape)

        VAE_out = self.VAE(inputs)

        self.feature_model = feature_model(inp_shape, num_feature_maps).f_model

        feature_maps = self.feature_model(VAE_out)
        img_generated = self.VAE.output
        feature_maps.append(img_generated)   # appending the reconstruction to the feature maps list
                                                                    # produced by the feature model
        

        self.full_model = keras.Model(self.VAE.input, feature_maps, name='full_model')
        self.output = self.full_model.output

    def summarise(self):
        self.full_model.summary()

