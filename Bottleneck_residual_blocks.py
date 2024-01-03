"""
@author: sourav
"""

"""
Module for the Bottleneck Residual Blocks and associated layers. Written in keras.

We shall use keras functional API (instead of using Sequential).

ReLU is always used as the activation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers



class conv2d_block(layers.Layer):     # it's not the most general conv2d layer, but it's what we need.

	def __init__(self, num_filters, kernel, strides= (1,1), padding='valid', kernel_initializer, use_bn='True', activation):
		super().__init__()
		
		self.conv2d = layers.Conv2D(filters=num_filters, kernel_size=kernel, strides=strides, 
								padding=padding, kernel_initializer=kernel_initializer)
								
	
	def call(self,x):
		x = self.conv2d(x)
		
		if use_bn=='True'
			x = layers.BatchNormalization(axis=-1)(x)
			
		x = layers.ReLU(x)
		
		return x
		
		
		
		
		
		
			
