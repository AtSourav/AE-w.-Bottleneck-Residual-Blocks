"""
@author: sourav
"""

"""
Module for the Bottleneck Residual Blocks and associated layers. Written in keras.

We shall use keras functional API (instead of using Sequential).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers



class conv2d_block(layers.Layer):
	def __init__(self,)
