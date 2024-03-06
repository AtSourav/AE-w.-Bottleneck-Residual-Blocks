"""
@author: sourav

Module for the Residual Blocks and associated layers. Written in keras.

We shall use keras functional API (instead of using Sequential).

ReLU is always used as the activation.

The input tensors should always be in the format with channels last.
"""



import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers






class conv2d_block(layers.Layer):     # it's not the most general conv2d layer we could use.

	def __init__(self, num_filters, kernel, kernel_initializer, strides= (1,1), padding='valid', use_bn='True', kernel_regularizer=None, *args, **kwargs):
		super().__init__()
		
		self.conv2d = layers.Conv2D(filters=num_filters, kernel_size=kernel, strides=strides, 
								padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
								
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

	def __init__(self, num_filters, kernel, kernel_initializer, strides= (1,1), padding='valid', use_bn='True', kernel_regularizer=None, *args, **kwargs):
		super().__init__()
		
		self.conv2dtrans = layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel, strides=strides, 
								padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
								
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
	
	def __init__(self, pool_size, strides=(1,1), padding='valid', *args, **kwargs):
		super().__init__()
		
		self.maxpool = layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
	
	def call(self,x):
		x_int = -x
		x_int = self.maxpool(x_int)
		x_out = -x_int
	
		return x_out
		
		
		
###########################################################################################################################################
		
		
		
class residual_conv2D_block(layers.Layer):

	def __init__(self, num_filters, num_layers, kernel, kernel_initializer, pooling, strides=(1,1), padding='valid', use_bn='True', kernel_regularizer=None, *args, **kwargs):
	
		# num_filters should be the same as the number of channels in the output of the previous layer, otherwise addition with the 
		# skipped connection won't be possible.
		
		# pooling can be 'min', 'max', or 'avg'.
		
		# num_layers is the number of convolutional layers skipped in the block, will take either 1 or 2
		
		super().__init__()
		
		self.conv_bn = conv2d_block(num_filters, kernel=kernel, kernel_initializer=kernel_initializer, strides=strides, padding=padding, use_bn=use_bn, kernel_regularizer=kernel_regularizer)
		
		self.conv_wobn = layers.Conv2D(filters=num_filters, kernel_size = kernel, kernel_initializer=kernel_initializer, strides=strides, padding=padding,
																	 kernel_regularizer=kernel_regularizer)
		
		# optional pooling to be applied to the skipped connection to ensure that the tensors to be added have 
		# equal dimension. this is only needed if the padding is 'valid'.
		
		self.minpool = min_pool2D(pool_size=kernel, strides=strides, padding=padding)
		self.maxpool = layers.MaxPooling2D(pool_size=kernel, strides=strides, padding=padding)
		self.avgpool = layers.AveragePooling2D(pool_size=kernel, strides=strides, padding=padding)
		
		self.batchnorm = layers.BatchNormalization(axis=-1)
		
		self.add = layers.Add()
		self.relu = layers.ReLU()
		
		self.padding = padding
		self.pooling = pooling
		self.num_layers = num_layers
		self.use_bn = use_bn
		
		
	def call(self,x):
		
		if self.num_layers != 1 and self.num_layers != 2:
			raise Exception("number of conv layers skipped must be either 1 or 2.")
			
		if self.num_layers == 1:
			x_int = self.conv_wobn(x)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
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
		else:
			x_int = self.conv_bn(x)
			x_int = self.conv_wobn(x_int)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
				if self.pooling != 'min' and self.pooling != 'max' and self.pooling != 'avg':
					raise Exception("pooling must be 'min', 'max', or 'avg'.")
				
				if self.pooling == 'min':
					x_skip = self.minpool(x)
					x_skip = self.minpool(x_skip)
				elif self.pooling == 'max':
					x_skip = self.maxpool(x)
					x_skip = self.maxpool(x_skip)
				else:
					x_skip = self.avgpool(x)
					x_skip = self.avgpool(x_skip)
			else:
				x_skip = x
		
			
			
		x_out = self.add([x_int, x_skip])
		
		if self.use_bn:
			x_out = self.batchnorm(x_out)
		
		x_out = self.relu(x_out)
		
		return x_out
		
		
		
##########################################################################################################################################		
		
		
		
		
class bridge_residual_conv2D_block(layers.Layer):                  # this block is to be used while changing the number of channels.
										# we shall add a unit convolution to the skipped connection to change the number of channels

	def __init__(self, num_filters, num_layers, kernel, kernel_initializer, pooling, strides=(1,1), padding='valid', use_bn='True', kernel_regularizer=None, *args, **kwargs):
	
		# num_filters should be the same as the number of channels in the output of the previous layer, otherwise addition with the 
		# skipped connection won't be possible.
		
		# pooling can be 'min', 'max', or 'avg'.
		
		# num_layers is the number of convolutional layers skipped in the block, will take either 1 or 2
		
		super().__init__()
		
		self.conv_bn = conv2d_block(num_filters, kernel=kernel, kernel_initializer=kernel_initializer, strides=strides, padding=padding, use_bn=use_bn, kernel_regularizer=kernel_regularizer)
		
		self.conv_wobn = layers.Conv2D(filters=num_filters, kernel_size = kernel, kernel_initializer=kernel_initializer, strides=strides, padding=padding,
																				kernel_regularizer=kernel_regularizer)
		
		# a 1x1 conv layer for the skip connection to reduce the number of channels so it can be added back to the main flow
		
		self.conv_skip = layers.Conv2D(filters=num_filters, kernel_size = 1, kernel_initializer=kernel_initializer)     # no regularization on this one
		
		# optional pooling to be applied to the skipped connection to ensure that the tensors to be added have 
		# equal dimension. this is only needed if the padding is 'valid'.
		
		self.minpool = min_pool2D(pool_size=kernel, strides=strides, padding=padding)
		self.maxpool = layers.MaxPooling2D(pool_size=kernel, strides=strides, padding=padding)
		self.avgpool = layers.AveragePooling2D(pool_size=kernel, strides=strides, padding=padding)
		
		self.batchnorm = layers.BatchNormalization(axis=-1)
		
		self.add = layers.Add()
		self.relu = layers.ReLU()
		
		self.padding = padding
		self.pooling = pooling
		self.num_layers = num_layers
		self.use_bn = use_bn
		
		
	def call(self,x):
		
		if self.num_layers != 1 and self.num_layers != 2:
			raise Exception("number of conv layers skipped must be either 1 or 2.")
			
		if self.num_layers == 1:
			x_int = self.conv_wobn(x)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
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
		else:
			x_int = self.conv_bn(x)
			x_int = self.conv_wobn(x_int)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
				if self.pooling != 'min' and self.pooling != 'max' and self.pooling != 'avg':
					raise Exception("pooling must be 'min', 'max', or 'avg'.")
				
				if self.pooling == 'min':
					x_skip = self.minpool(x)
					x_skip = self.minpool(x_skip)
				elif self.pooling == 'max':
					x_skip = self.maxpool(x)
					x_skip = self.maxpool(x_skip)
				else:
					x_skip = self.avgpool(x)
					x_skip = self.avgpool(x_skip)
			else:
				x_skip = x
		
		
				
		x_skip = self.conv_skip(x_skip)	
			
		x_out = self.add([x_int, x_skip])
		
		if self.use_bn:
			x_out = self.batchnorm(x_out)
		
		x_out = self.relu(x_out)
		
		return x_out
		
		
		
		
##########################################################################################################################################
		
		
		
		
class residual_conv2Dtrans_block(layers.Layer):

	def __init__(self, num_filters, num_layers, kernel, kernel_initializer, padding='valid', use_bn='True', kernel_regularizer=None, *args, **kwargs):
	
		# num_filters should be the same as the number of channels in the output of the previous layer, otherwise addition with the 
		# skipped connection won't be possible.
		
		# we are defining this with a fixed stride length of 1.
		
		# num_layers is the number of conv trans layers skipped, this is either 1 or 2
		
		super().__init__()
		
		self.convtrans_bn = conv2dtrans_block(num_filters, kernel=kernel, kernel_initializer=kernel_initializer, padding = padding, use_bn=use_bn, kernel_regularizer=kernel_regularizer)
		
		self.convtrans_wobn = layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel, kernel_initializer=kernel_initializer, padding=padding, kernel_regularizer=kernel_regularizer)
		
		# optional zero padding to be applied to the skipped connection to ensure that the tensors to be added have 
		# equal dimension. this is only needed if the padding is 'valid'.
		
		p = (kernel - 1)//2
		self.zeropad_sym = layers.ZeroPadding2D(padding=p)
		self.zeropad_asym = layers.ZeroPadding2D(padding=((p,kernel-1-p),(p,kernel-1-p)))
		self.doublezeropad = layers.ZeroPadding2D(padding= kernel-1)
		
		self.add = layers.Add()
		self.relu = layers.ReLU()
		self.batchnorm = layers.BatchNormalization(axis=-1)
		
		self.kernel = kernel
		self.padding = padding
		self.num_layers = num_layers
		self.use_bn = use_bn
		
		
	def call(self,x):
		
		if self.num_layers != 1 and self.num_layers != 2:
			raise Exception("number of conv layers skipped must be either 1 or 2.")
			
		if self.num_layers == 1:
			x_int = self.convtrans_wobn(x)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
			
				if (self.kernel - 1)%2 == 0:                              #  in this case the padding can be symmetric
					x_skip = self.zeropad_sym(x)        	     	     # to make sure x_skip has the same 
										     		# dimensions as x_int
				else:
					x_skip = self.zeropad_asym(x) 			# in this case the zero padding is assymetric
				
			else:
				x_skip = x
		else:
			x_int = self.convtrans_bn(x)
			x_int = self.convtrans_wobn(x_int)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
				x_skip = self.doublezeropad(x)				# in this case the padding is always symmetric
			else:
				x_skip = x
		
		
		x_out = self.add([x_int, x_skip])
		
		if self.use_bn:
			x_out = self.batchnorm(x_out)
		
		x_out = self.relu(x_out)
		
		return x_out



#########################################################################################################################




class bridge_residual_conv2Dtrans_block(layers.Layer):			# this block is to be used while changing the number of channels.
										# we shall add a unit convolution to the skipped connection to change the number of channels


	def __init__(self, num_filters, num_layers, kernel, kernel_initializer, padding='valid', use_bn='True', kernel_regularizer=None, *args, **kwargs):
	
		# num_filters should be the same as the number of channels in the output of the previous layer, otherwise addition with the 
		# skipped connection won't be possible.
		
		# we are defining this with a fixed stride length of 1.
		
		# num_layers is the number of conv trans layers skipped, this is either 1 or 2
		
		super().__init__()
		
		self.convtrans_bn = conv2dtrans_block(num_filters, kernel=kernel, kernel_initializer=kernel_initializer, padding = padding, use_bn=use_bn, kernel_regularizer=kernel_regularizer)
		
		self.convtrans_wobn = layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel, kernel_initializer=kernel_initializer, padding=padding,
																			kernel_regularizer=kernel_regularizer)
		
		# a 1x1 conv layer for the skip connection to reduce the number of channels so it can be added back to the main flow
		
		self.convtrans_skip = layers.Conv2DTranspose(filters=num_filters, kernel_size = 1, kernel_initializer=kernel_initializer)     # no regularization on this one
		
		# optional zero padding to be applied to the skipped connection to ensure that the tensors to be added have 
		# equal dimension. this is only needed if the padding is 'valid'.
		
		p = (kernel - 1)//2
		self.zeropad_sym = layers.ZeroPadding2D(padding=p)
		self.zeropad_asym = layers.ZeroPadding2D(padding=((p,kernel-1-p),(p,kernel-1-p)))
		self.doublezeropad = layers.ZeroPadding2D(padding= kernel-1)
		
		self.add = layers.Add()
		self.relu = layers.ReLU()
		self.batchnorm = layers.BatchNormalization(axis=-1)
		
		self.kernel = kernel
		self.padding = padding
		self.num_layers = num_layers
		self.use_bn = use_bn
		
		
	def call(self,x):
		
		if self.num_layers != 1 and self.num_layers != 2:
			raise Exception("number of conv layers skipped must be either 1 or 2.")
			
		if self.num_layers == 1:
			x_int = self.convtrans_wobn(x)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
			
				if (self.kernel - 1)%2 == 0:                              #  in this case the padding can be symmetric
					x_skip = self.zeropad_sym(x)        	     	     # to make sure x_skip has the same 
										     		# dimensions as x_int
				else:
					x_skip = self.zeropad_asym(x) 			# in this case the zero padding is assymetric
				
			else:
				x_skip = x
		else:
			x_int = self.convtrans_bn(x)
			x_int = self.convtrans_wobn(x_int)
			
			if self.padding != 'valid' and self.padding != 'same':
				raise Exception("padding must be either 'valid' or 'same'.")
		
			if self.padding =='valid':
				x_skip = self.doublezeropad(x)				# in this case the padding is always symmetric
			else:
				x_skip = x
				
		x_skip = self.convtrans_skip(x_skip)
		
		
		x_out = self.add([x_int, x_skip])
		
		if self.use_bn:
			x_out = self.batchnorm(x_out)
		
		x_out = self.relu(x_out)
		
		return x_out
		
		
