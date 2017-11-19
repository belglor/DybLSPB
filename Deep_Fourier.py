from __future__ import absolute_import, division, print_function 

#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
# import sklearn.datasets
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join('.', '..')) 
import utils # -*- coding: utf-8 -*-


###TWO LAYER CNN###
tf.reset_default_graph()

#Network dimension:
#    - Input data (k0 x 1) vector;
#    - Conv1 kernel with phi0 filters;
#    - After MaxPool1, (k1 x phi0) data where k1 depends on kernel size, strides, maxpool;
#    - Conv2 kernel with phi1 = 60 filters;
#    - After MaxPool1, have (41 x 60) data;
#    

# Input data size
length = 20992; #k0 
nchannels = 1; #??? Maybe two because stereo audiofiles?

# Lyers hyperparameters definitions (NB: they've been calculated keeping
# the proportions with the dimension of the data from the other papers, i.e.
# 15:(2200Hz*310ms)~=155:(20992)

# Conv1, MaxPool1 parameters
padding_conv1 = "valid"
filters_1 = 80 #phi1
kernel_size_1 = 155
strides_conv1 = 10
padding_pool1 = "valid"
pool_size_1 = 9
strides_pool1 = 3

# Conv2, MaxPool2 parameters
padding_conv2 = 'valid'
filters_2 = 60 #phi2
kernel_size_2 = 15
strides_conv2 = 4
padding_pool2 = 'valid'
pool_size_2 = 9
strides_pool2 = 4

x_pl_1 = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')


### I MODIFIED CONV2D TO CONV1D, CHECK PARAMETERS
### ALSO, SWITCH TO TF.CONV1D INSTEAD OF KERAS.LAYERS

print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')

with tf.variable_scope('convLayer1'):
    ### INPUT DATA
    print('x_pl_1 \t\t', x_pl_1.get_shape());
    
    ### CONV1 LAYER
    # Layer build
    z1 = tf.layers.conv1d(   inputs=x_pl_1,
                             filters=filters_1,
                             kernel_size=kernel_size_1,
                             strides=strides_conv1,
                             padding=padding_conv1,
                             #data_format='channels_last',
                             #dilation_rate=1,
                             activation=tf.nn.relu,
                             use_bias=True,
                             #kernel_initializer=None,
                             bias_initializer=tf.zeros_initializer(),
                             #kernel_regularizer=None,
                             #bias_regularizer=None,
                             #activity_regularizer=None,
                             #kernel_constraint=None,
                             #bias_constraint=None,
                             trainable=True,
                             name="conv_1",
                             #reuse=None
                            )
    # Input pass, activation 
    print('conv1 \t\t', z1.get_shape())                     
                            
    ### MAX_POOL1
    # For tf v1.3, use max_pooling1d instead of MaxPooling1D which is for v1.4
    # works the same, even same arguments, but MaxPooling1D creates a layer that is 
    # later applied on z1, while max_pooling1d takes the conv layer data as argument too
    a1 = tf.layers.max_pooling1d(    inputs = z1,
                                    pool_size=pool_size_1,
                                    strides=strides_pool1,
                                    padding=padding_pool1, 
                                    name='pool_1'
                                )
                                   
    # Activation pass, pooling
    #a1 = (pool1(z1));
    print('pool1 \t\t', a1.get_shape())
    
with tf.variable_scope('convLayer2'):    
    ### CONV2 LAYER
    # Layer build
    z2 = tf.layers.conv1d(   inputs=a1,
                             filters=filters_2,
                             kernel_size=kernel_size_2,
                             strides=strides_conv2,
                             padding=padding_conv2,
                             #data_format='channels_last',
                             #dilation_rate=1,
                             activation=tf.nn.relu,
                             use_bias=True,
                             #kernel_initializer=None,
                             bias_initializer=tf.zeros_initializer(),
                             #kernel_regularizer=None,
                             #bias_regularizer=None,
                             #activity_regularizer=None,
                             #kernel_constraint=None,
                             #bias_constraint=None,
                             trainable=True,
                             name="conv_2",
                             #reuse=None
                            )
    # Input pass, activation 
    print('conv2 \t\t', z2.get_shape())                     
                            
    ### MAX_POOL2
    a2 = tf.layers.max_pooling1d(   inputs  = z2,
                                    pool_size=pool_size_2,
                                    strides=strides_pool2,
                                    padding=padding_pool2, 
                                    name='pool_2'
                                )
                                   
    # Activation pass, pooling
    #a2 = (pool2(z2));
    print('pool2 \t\t', a2.get_shape())

print('Model consits of ', utils.num_params(), 'trainable parameters.')