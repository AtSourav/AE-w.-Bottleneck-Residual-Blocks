"""
@author: sourav

We define the loss functions and write the training loop in this module. 
"""

import numpy as np
import tensorflow as tf
from tensorflow import image
from tensorflow import keras
from tensorflow import math

from keras import losses
import losses.MeanSquaredError as MSE

from keras import layers
from keras import utils
from keras import metrics
from keras import backend as K
from keras import initialiazers


import os 
import random
import matplotlib.pyplot as plt

###########################################################################################################################
# defining the CB loss function

def norm_CB(z, l_cutoff = 0.495, u_cutoff = 0.505):                                 

    gate = math.logical_and(math.greater(z,l_cutoff), math.greater(u_cutoff,z))

    z = tf.clip_by_value(z, clip_value_min = K.epsilon(), clip_value_max = 1 - K.epsilon())     


    norm_reg = (2*math.atanh(1 - 2*z_reg))/(1 - 2*z_reg)         
    norm_taylor = 2.0 + (8.0/3.0)*math.pow(z-0.5,2) + (32.0/5.0)*math.pow(z-0.5,4)  +  (128.0/7.0)*math.pow(z-0.5,6)          


    norm = tf.where(gate, norm_taylor, norm_reg)          

    return norm

@tf.function
def CB_logloss(true, pred):
  true = layers.Flatten()(true)
  pred = layers.Flatten()(pred)
  bce = losses.binary_crossentropy(true,pred)

  corrected_loss_tensor = bce - tf.reduce_mean(math.log(norm_CB(pred)), axis=-1 )    

  return tf.reduce_mean(corrected_loss_tensor)


##########################################################################################################################
# SSIM loss

def SSIMloss(true,pred):
   
   return 1 - image.ssim(true,pred,1.0)
   

##########################################################################################################################




