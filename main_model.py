"""
@author: sourav

We define the main models as classes. 
"""

# when custom layers/models are built in keras by subclassing, we cannot infer input/output 
# before the model any real data is passed into the model.
# This is different from when the models are defined using the Functional API or using Sequential.
# Don't understand the inner details, but apparently, this is the case. 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import initializers
from keras import layers
from keras.layers import Layer
from keras.models import Model

# from Residual_blocks import *
from Residual_blocks2 import *



class Encoder(Layer):
    '''
    Returns an Encoder as a Layer object. The encoder is composed of convolutional residual blocks and 
    a few dense layers on top. 
    --Add the architecture.  
    '''

    def __init__(self, initializer, latent_dim, *args, **kwargs):
        super().__init__()

        self.layers_list = [bridge_residual_conv2D_block(64, 2, 3, initializer, 'min', name='bres_1'),
                           
                    bridge_residual_conv2D_block(128, 2, 3, initializer, 'min', name='bres_2'),

                    residual_conv2D_block(128, 2, 3, initializer, 'min', padding = 'same', name='res_3'),

                    residual_conv2D_block(128, 1, 1, initializer, 'min', name='res_4'),
        
                    layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='pool_5'),

                    bridge_residual_conv2D_block(256, 2, 3, initializer, 'min', name='bres_6'),

                    residual_conv2D_block(256, 1, 1, initializer, 'min', name='res_7'),

                    residual_conv2D_block(256, 2, 3, initializer, 'min', padding = 'same', name='res_8'),

                    residual_conv2D_block(256, 1, 1, initializer, 'min', name='res_9'),

                    layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", name='pool_10'),

                    bridge_residual_conv2D_block(512, 2, 3, initializer, 'min', padding = 'same', name='bres_11'),

                    residual_conv2D_block(512, 1, 1, initializer, 'min', name='res_12'),

                    residual_conv2D_block(512, 2, 3, initializer, 'min', padding = 'same', name='res_13'),

                    residual_conv2D_block(512, 1, 1, initializer, 'min', name='res_14'),

                    layers.Flatten(name='flatten'),

                    layers.Dense(2*latent_dim, name='dense_15'),

                    layers.ReLU(name='relu_16')
                    
        ]

        self.z_out = layers.Dense(latent_dim, name="z_out")

    def call(self, input):

        x = input
        for layer in self.layers_list:
            x = layer(x)

        z_out = self.z_out(x)

        return z_out


##############################################################################################
    

class Sampling(Layer):

    def __init__(self, seed):
        super().__init__()

        self.seed = seed


    def call(self, arg):
        z_m, z_log_v = arg
        batch = tf.shape(z_m)[0]
        dim = tf.shape(z_m)[1]
        eps = tf.random.normal(shape=(batch,dim), seed=self.seed)
        return z_m + tf.exp(0.5*z_log_v)*eps

    

class Encoder_VAE(Encoder):
    '''
    Encoder for a VAE, subclassed from the encoder for an AE.
    '''

    def __init__(self, initializer, latent_dim, seed, *args, **kwargs):
        super().__init__(initializer, latent_dim)
        # repeating the arguments can be avoided by the use of *args **kwargs in both __init
        # and super().__init__ 

        self.latent_dim = latent_dim
        self.seed = seed

    def build(self,input_shape):

        self.z_mean = layers.Dense(self.latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(self.latent_dim, name='z_log_var')
        self.sampling = Sampling(self.seed)
        super(Encoder_VAE, self).build(input_shape)         # to build automatically when the class is instantiated?
                                                            # so we do not have to call the build() method?
    
    def call(self, input): 

        x = input
        for layer in self.layers_list:
            x = layer(x)
        z_m = self.z_mean(x)
        z_lv = self.z_log_var(x)

        z = self.sampling([z_m, z_lv])

        return [z_m, z_lv, z]



################################################################################################
    

class Decoder(Layer):
    '''
    Returns a Decoder as a Layer object. The decoder is composed of convolutional residual blocks and 
    a few dense layers on top. 
    --Add the architecture.  
    '''

    def __init__(self, initializer, latent_dim, *args, **kwargs):
        super().__init__()

        self.layers_list = [layers.Dense(2*latent_dim, name='dense_1'), 
                    
                    layers.ReLU(name='relu_2'),

                    layers.Dense(4*latent_dim, name='dense_3'),

                    layers.ReLU(name='relu_4'),

                    layers.Dense(2*2*1024, name='dense_5'),

                    layers.ReLU(name='relu_6'),

                    layers.Reshape((2,2,1024), name='reshape'),

                    bridge_residual_conv2Dtrans_block(1024, 1, 1, initializer, name='bres_7'),

                    bridge_residual_conv2Dtrans_block(512, 1, 1, initializer, name='bres_8'),

                    residual_conv2Dtrans_block(512, 2, 3, initializer, padding='same', name='res_9'),

                    bridge_residual_conv2Dtrans_block(512, 1, 1, initializer, name='bres+10'),

                    residual_conv2Dtrans_block(512, 2, 3, initializer, name='res_11'),

                    bridge_residual_conv2Dtrans_block(256, 1, 1, initializer, name='bres_12'),

                    residual_conv2Dtrans_block(256, 2, 3, initializer, padding='same', name='res_13'),

                    residual_conv2Dtrans_block(256, 1, 1, initializer, name='res_14'),

                    residual_conv2Dtrans_block(256, 2, 3, initializer, name='res_15'),

                    layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear', name='upsampling_16'),

                    bridge_residual_conv2Dtrans_block(128, 2, 3, initializer, padding='same', name='bres_17'),

                    residual_conv2Dtrans_block(128, 2, 3, initializer, padding='same', name='res_18'),

                    layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear', name='upsampling_19'),

                    bridge_residual_conv2D_block(128, 2, 3, initializer, 'min', name='bres_20'),

                    residual_conv2D_block(128, 1, 1, initializer, 'min', name='res_21'),

                    bridge_residual_conv2D_block(64, 1, 2, initializer, 'min', name='res_22'),

                    residual_conv2D_block(64, 1, 2, initializer, 'min', name='res_23'),

                    bridge_residual_conv2D_block(32, 1, 2, initializer, 'min', name='bres_24'),

                    residual_conv2D_block(32, 1, 2, initializer, 'min', name='res_25')
        ]

        self.img_out = layers.Conv2D(3, 1, activation='sigmoid', padding='valid', kernel_initializer=initializer, name='img_out')

    def call(self, latent_input):

        x = latent_input
        for layer in self.layers_list:
            x = layer(x)
        
        img_out = self.img_out(x)

        return img_out
    

#########################################################################################################################
    



class Decoder_VAE(Decoder):
    '''
    Decoder for a VAE, subclassed from the decoder for an AE, identical to it.
    '''

    def __init__(self, initializer, latent_dim, *args, **kwargs):
        super().__init__(initializer, latent_dim)

    def call(self, latent_inp):

        x = latent_inp
        for layer in self.layers_list:
            x = layer(x)

        img_out = self.img_out(x)

        return img_out
    


###########################################################################################################################
    



class AE(Model):
    '''
    Returns an AE subclassed from the Model class, composed of an Encoder object and a Decoder object.
    '''

    def __init__(self, initializer, latent_dim, *args, **kwargs):
        super().__init__()

        self.encoder = Encoder(initializer, latent_dim)
        self.decoder = Decoder(initializer, latent_dim)

    def call(self, input):

        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return decoded
    


###########################################################################################################################
    


class VAE(Model):
    '''
    Returns a VAE subclassed from the Model class, composed of Encoder_VAE and Decoder_VAE.
    '''

    def __init__(self,initializer, latent_dim, seed, *args, **kwargs):
        super().__init__()

        self.encoder = Encoder_VAE(initializer, latent_dim, seed)
        self.decoder = Decoder_VAE(initializer, latent_dim)

    def call(self, input):

        encoded = self.encoder(input)
        decoded = self.decoder(encoded[2])

        return decoded





