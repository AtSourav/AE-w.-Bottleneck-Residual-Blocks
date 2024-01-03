"""
@author: sourav
"""

"""
Module for the Bottleneck Residual Blocks and associated layers. Written in keras.

We shall use keras functional API (instead of using Sequential).

ReLU is always used as the activation.

The input tensors should always be in the format with channels last.
"""



import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers



class conv2d_block(layers.Layer):     # it's not the most general conv2d layer we could use.

	def __init__(self, num_filters, kernel, strides= (1,1), padding='valid', kernel_initializer, use_bn='True'):
		super().__init__()
		
		self.conv2d = layers.Conv2D(filters=num_filters, kernel_size=kernel, strides=strides, 
								padding=padding, kernel_initializer=kernel_initializer)
								
	
	def call(self,x):
		x = self.conv2d(x)
		
		if use_bn=='True'
			x = layers.BatchNormalization(axis=-1)(x)
			
		x = layers.ReLU()(x)
		
		return x
		
		
def min_pool2D(pool_size, strides, padding='valid',x):
	
	x_int = -x
	x_int = layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(x_int)
	x_out = -x_int
	
	return x_out
		
		
		
		
class bottleneck_residual_block(layers.Layer):

	def __init__(self, num_filters, kernel, strides=(1,1), padding='valid', kernel_initializer, use_bn='True', pooling):
	
		# kernel, strides, and padding are parameters only relevant to the second conv2d layer that is sandwiched 
		# between the two 1x1 kernel conv layers.
		
		# pooling can be 'min', 'max', or 'avg'.
		
		super().__init__()
		
		# first conv2d block with 1x1 kernel and strides = (1,1)
		self.conv1 = conv2dblock(num_filters//2, kernel=1, kernel_initializer=kernel_initializer, use_bn=use_bn)
		
		# second conv2d layer with kernel size and strides as specified
		self.conv2 = conv2dblock(num_filters//2, kernel=kernel, strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bn=use_bn)
		
		# third conv2d layer with 1x1 kernel to restore the number of channels. No activation after this one.
		self.conv3 = layers.Conv2D(filters=num_filters, kernel_size=1, kernel_initializer=kernel_initializer)
		
		# optional min pooling to be applied to the skipped connection to ensure that the tensors to be added have 
		# equal dimension. this is only needed if the padding is 'valid'.
		
		self.kernel = kernel
		self.strides = strides
		self.padding = padding
		self.pooling = pooling
		
		
	def call(self,x):
	
		x_int = self.conv1(x)
		x_int = self.conv2(x_int)
		x_int = self.conv3(x_int)
		
		if self.padding != 'valid' and self.padding != 'same':
			raise Exception("padding must be either 'valid' or 'same'.")
		
		if self.padding=='valid':
			if self.pooling != 'min' and self.pooling != 'max' and self.pooling != 'avg':
				raise Exception("pooling must be 'min', 'max', or 'avg'.")
				
			if self.pooling == 'min':
				x_skip = min_pool2D(pool_size=self.kernel, strides=self.strides, padding=self.padding, x)
			elif self.pooling == 'max':
				x_skip = layers.MaxPooling2D(pool_size=self.kernel, strides=self.strides, padding=self.padding)(x)
			else:
				x_skip = layers.AveragePooling2D(pool_size=self.kernel, strides=self.strides, padding=self.padding)(x)
				
		else:
			x_skip = x
			
			
		x_out = layers.Add()[x_int, x_skip]
		
		x_out = layers.ReLU()(x_out)
		
		return x_out
		
		
		
		
		
		
			
