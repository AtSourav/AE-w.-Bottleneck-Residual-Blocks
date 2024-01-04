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

	def __init__(self, num_filters, kernel, kernel_initializer, strides= (1,1), padding='valid', use_bn='True'):
		super().__init__()
		
		self.conv2d = layers.Conv2D(filters=num_filters, kernel_size=kernel, strides=strides, 
								padding=padding, kernel_initializer=kernel_initializer)
								
		self.batchnorm = layers.BatchNormalization(axis=-1)
		
		self.relu = layers.ReLU()
								
		self.use_bn = use_bn
								
	
	def call(self,x):
		x = self.conv2d(x)
		
		if self.use_bn=='True':
			x = self.batchnorm(x)
			
		x = self.relu(x)
		
		return x
		
		
class conv2dtrans_block(layers.Layer):    

	def __init__(self, num_filters, kernel, kernel_initializer, strides= (1,1), padding='valid', use_bn='True'):
		super().__init__()
		
		self.conv2dtrans = layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel, strides=strides, 
								padding=padding, kernel_initializer=kernel_initializer)
								
		self.batchnorm = layers.BatchNormalization(axis=-1)
		
		self.relu = layers.ReLU()
		
		self.use_bn = use_bn	
							
	
	def call(self,x):
		x = self.conv2dtrans(x)
		
		if self.use_bn=='True':
			x = self.batchnorm(x)
			
		x = self.relu(x)
		
		return x
		
		
class min_pool2D(layers.Layer):
	
	def __init__(self, pool_size, strides=(1,1), padding='valid'):
		super().__init__()
		
		self.maxpool = layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
	
	def call(self,x):
		x_int = -x
		x_int = self.maxpool(x_int)
		x_out = -x_int
	
		return x_out
		
		
		
		
class bottleneck_residual_conv2D_block(layers.Layer):

	def __init__(self, num_filters, compress_ratio, kernel, kernel_initializer, pooling, strides=(1,1), padding='valid', use_bn='True'):
	
		# num_filters should be the same as the number of channels in the output of the previous layer, otherwise addition with the 
		# skipped connection won't be possible.
	
		# kernel, strides, and padding are parameters only relevant to the second conv2d layer that is sandwiched 
		# between the two 1x1 kernel conv layers.
		
		# pooling can be 'min', 'max', or 'avg'.
		
		# compress_ratio is the ratio by which we compress using the first 1x1 conv layer.
		# using a compress_ratio of less than 1 will create an inverted residual block.
		
		super().__init__()
		
		# first conv2d block with 1x1 kernel and strides = (1,1)
		self.conv1 = conv2d_block(num_filters//compress_ratio, kernel=1, kernel_initializer=kernel_initializer, use_bn=use_bn)
		
		# second conv2d layer with given kernel size and strides as specified
		self.conv2 = conv2d_block(num_filters//compress_ratio, kernel=kernel, kernel_initializer=kernel_initializer, 
										strides=strides, padding=padding, use_bn=use_bn)
		
		# third conv2d layer with 1x1 kernel to restore the number of channels. No activation after this one.
		self.conv3 = layers.Conv2D(filters=num_filters, kernel_size=1, kernel_initializer=kernel_initializer)
		
		# optional min pooling to be applied to the skipped connection to ensure that the tensors to be added have 
		# equal dimension. this is only needed if the padding is 'valid'.
		
		self.minpool = min_pool2D(pool_size=kernel, strides=strides, padding=padding)
		self.maxpool = layers.MaxPooling2D(pool_size=kernel, strides=strides, padding=padding)
		self.avgpool = layers.AveragePooling2D(pool_size=kernel, strides=strides, padding=padding)
		
		self.add = layers.Add()
		self.relu = layers.ReLU()
		
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
				x_skip = self.minpool(x)
			elif self.pooling == 'max':
				x_skip = self.maxpool(x)
			else:
				x_skip = self.avgpool(x)
				
		else:
			x_skip = x
			
			
		x_out = self.add([x_int, x_skip])
		
		x_out = self.relu(x_out)
		
		return x_out
		
		
		
		

class bottleneck_residual_conv2Dtrans_block(layers.Layer):

	def __init__(self, num_filters, compress_ratio, kernel, kernel_initializer, padding='valid', use_bn='True'):
	
		# num_filters should be the same as the number of channels in the output of the previous layer, otherwise addition with the 
		# skipped connection won't be possible.
	
		# kerneland padding are parameters only relevant to the second conv2d layer that is sandwiched 
		# between the two 1x1 kernel conv layers.
		
		# we are defining this with a fixed stride length of 1.
		
		# pooling can be 'min', 'max', or 'avg'.
		
		# compress_ratio is the ratio by which we compress using the first 1x1 conv layer.
		# using a compress_ratio of less than 1 will create an inverted residual block.
		
		super().__init__()
		
		# first conv2dtrans block with 1x1 kernel and strides = (1,1)
		self.conv1trans = conv2dtrans_block(num_filters//compress_ratio, kernel=1, kernel_initializer=kernel_initializer, use_bn=use_bn)
		
		# second conv2dtrans layer with given kernel size and strides = (1,1)
		self.conv2trans = conv2dtrans_block(num_filters//compress_ratio, kernel=kernel, kernel_initializer=kernel_initializer, 
														padding=padding, use_bn=use_bn)
		
		# third conv2dtrans layer with 1x1 kernel to restore the number of channels. No activation after this one.
		self.conv3trans = layers.Conv2DTranspose(filters=num_filters, kernel_size=1, kernel_initializer=kernel_initializer)
		
		# optional min pooling to be applied to the skipped connection to ensure that the tensors to be added have 
		# equal dimension. this is only needed if the padding is 'valid'.
		
		p = (kernel - 1)//2
		self.zeropad_sym = layers.ZeroPadding2D(padding=p)
		self.zeropad_asym = layers.ZeroPadding2D(padding=((p,kernel-1-p),(p,kernel-1-p)))
		
		self.add = layers.Add()
		self.relu = layers.ReLU()
		
		self.kernel = kernel
		self.padding = padding
		
		
	def call(self,x):
	
		x_int = self.conv1trans(x)
		x_int = self.conv2trans(x_int)
		x_int = self.conv3trans(x_int)
		
		if self.padding != 'valid' and self.padding != 'same':
			raise Exception("padding must be either 'valid' or 'same'.")
		
		if self.padding=='valid':
			if (self.kernel - 1)%2 == 0:                              #  in this case the padding can be symmetric
				x_skip = self.zeropad_sym(x)        	     	     # to make sure x_skip has the same 
										     # dimensions as x_int
			else:
				# in this case the zero padding is assymetric
				x_skip = self.zeropad_asym(x) 
		else:
			x_skip = x
			
			
		x_out = self.add([x_int, x_skip])
		
		x_out = self.relu(x_out)
		
		return x_out
		
		
		
		
		
		
			
