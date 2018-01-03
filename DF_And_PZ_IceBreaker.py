#####################
###   CODE SETUP  ###
#####################

from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join('.', '..'))
import utils
import scipy.io
import batch_loader as bl
import tensorflow as tf
import time
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.layers.pooling import MaxPooling1D
from sklearn.metrics import confusion_matrix
from icebreaker import *

###RESET GRAPH
tf.reset_default_graph()

PHASE1 = 1 #Train only DF 
PHASE2 = 2 #Train DF and first PZ layer
PHASE3 = 3 #Train everything
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DF_arch =   "MST" #"Heuri1"
STEP = PHASE1
good_old_PZ_W_file = "results_mat/trainedweights/Piczak.mat"
Spcgm_forcing_trained_W_file = "results_mat/trainedweights/MST_sf.mat"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

n_fold = 10
# Folds we validate and test on (only relevant if RUN_CV==False). Indices of the folds are in the "Matlab" naming mode
k_valid = 9
k_test = 10
# If we just want to test quickly on a few epochs
RUN_FAST = True
# If we want oversampled (balanced) mini-batches or not
BALANCED_BATCHES = False
# Learning rate
learning_rate = 0.002
# Number of epochs (only one is relevant acc. to which RUN_FAST value has been given)
max_epochs_fast = 2
max_epochs_regular = 100
# Batch size
batch_size = 1000
USE_GRADIENT_CLIPPING = False
STORE_GRADIENT_NORMS = False
USE_LR_DECAY = False #learning rate decay: exponentially decreasing the learning rate over the epochs, so that we do not overshoot the (local) optimum when we get close
learning_rate_END = .0001 #if USE_LR_DECAY is set to True, then the learning rate will exponentially decay at every epoch and will reach learning_rate_END at the last epoch
NAME_SAFETY_PREFIX = "" #set this to "" for normal behaviour. Set it to any other string if you are just testing things and want to avvoid overwriting important stuff
nskip_iter_GN = 20 #every nskip_iter_GN'th iteration we will take one measurement of the gradient L2 norm (measuring it at every iteration takes too much comp. time)  
#########################
###   PRE-WORK WORK!  ###
#########################

#######################################
### Complementary actions to perform before defining the network

# Folds and number of epochs
if RUN_FAST:
    max_epochs = max_epochs_fast
    print('----------------------DESCRIPTION-----------------------------')
    print('Fast mode')
    print('{0} epochs to be run'.format(max_epochs))
else:
    max_epochs = max_epochs_regular
    print('------')
    print('Regular mode')
    print('{0} epochs to be run'.format(max_epochs))

ib = icebreaker(STEP, learning_rate, max_epochs, DF_arch, good_old_PZ_W_file, Spcgm_forcing_trained_W_file)

print('"A" method with validation on fold' + str(k_valid) + 'and test on fold' + str(k_test))
if USE_LR_DECAY:
    LRD_FACTOR = (learning_rate_END/learning_rate)**(1./(max_epochs-1))
    lr_array = learning_rate*(LRD_FACTOR**np.arange(max_epochs))
else:
    lr_array = learning_rate*np.ones(max_epochs, float) # learning rate stays constant
print("the learning rate will start at {0} and end at {1}".format(lr_array[0], lr_array[-1]) )

# Naming of output files
# We just shift -1 the folds indices to match with Python way to think
k_valid = k_valid - 1
k_test = k_test - 1

momentum = .9

# create these 3 folders if you don't have them
# if you really have to change these folders, do it HERE and not further down in the code, and do not git push these folders
data_folder = "./data/"
result_mat_folder = "./results_mat/"

save_path_perf = result_mat_folder + "performance/"
save_path_numpy_weights = result_mat_folder + "trainedweights/"

for directory in [save_path_perf, save_path_numpy_weights]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# %%
################################
###   GET  TRAINED WEIGHTS   ###
################################
# Separate weights for DF

###GENERAL VARIABLES
# Input data size
length = 20992  # k0
nchannels = 1  # ??? Maybe two because stereo audiofiles?
num_classes = 10
#----------------------------------------
	###DEEP_FOURIER LAYERS HYPERPARAMETERS
#----------------------------------------

if DF_arch == "Heuri1":
	##############################
	###   NETWORK PARAMETERS   ###
	##############################

	# Conv1, MaxPool1 parameters
	DF_padding_conv1 = "valid"
	DF_filters_1 = 80  # phi1
	DF_kernel_size_1 = 155
	DF_strides_conv1 = 10
	DF_padding_pool1 = "valid"
	DF_pool_size_1 = 9
	DF_strides_pool1 = 3

	# Conv2, MaxPool2 parameters
	DF_padding_conv2 = 'valid'
	DF_filters_2 = 60  # phi2
	DF_kernel_size_2 = 15
	DF_strides_conv2 = 4
	DF_padding_pool2 = 'valid'
	DF_pool_size_2 = 9
	DF_strides_pool2 = 4

	###PLACEHOLDER VARIABLES
	x_pl = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')
	y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
	y_pl = tf.cast(y_pl, tf.float32)

	# %%
	##############################
	###   NETWORK DEFINITION   ###
	##############################

	print('Trace of the tensors shape as it is propagated through the network.')
	print('Layer name \t Output size')
	print('----------------------------')

	### DEEP FOURIER NETWORK
	with tf.variable_scope('DF_convLayer1'):
		### INPUT DATA
		print("--- Deep Fourier conv layer 1")
		print('x_pl \t\t', x_pl.get_shape())

		### CONV1 LAYER
		# Layer build
		### CHECK IF WEIGHT LOADING\INITIALIZE
		if ib.shall_DF_be_loaded():
			z1 = tf.layers.conv1d(inputs=x_pl,
								  filters=DF_filters_1,
								  kernel_size=DF_kernel_size_1,
								  strides=DF_strides_conv1,
								  padding=DF_padding_conv1,
								  # data_format='channels_last',
								  # dilation_rate=1,
								  activation=tf.nn.relu,
								  use_bias=True,
								  kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[0])),
								  bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[1])),
								  # kernel_regularizer=None,
								  # bias_regularizer=None,
								  # activity_regularizer=None,
								  # kernel_constraint=None,
								  # bias_constraint=None,
								  trainable=True,
								  name="DF_conv_1",
								  # reuse=None
								  )
			print('Pretrained DF_conv1d_1 loaded!')
		else:
			z1 = tf.layers.conv1d(inputs=x_pl,
								  filters=DF_filters_1,
								  kernel_size=DF_kernel_size_1,
								  strides=DF_strides_conv1,
								  padding=DF_padding_conv1,
								  # data_format='channels_last',
								  # dilation_rate=1,
								  activation=tf.nn.relu,
								  use_bias=True,
								  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
																						  dtype=tf.float32),
								  bias_initializer=tf.zeros_initializer(),
								  # kernel_regularizer=None,
								  # bias_regularizer=None,
								  # activity_regularizer=None,
								  # kernel_constraint=None,
								  # bias_constraint=None,
								  trainable=True,
								  name="DF_conv_1",
								  # reuse=None
								  )
			print('DF_conv1d_1 reinitialized!')

		# Input pass, activation
		print('DF_conv1 \t\t', z1.get_shape())

		### MAX_POOL1
		pool1 = MaxPooling1D(pool_size=DF_pool_size_1,
							 strides=DF_strides_pool1,
							 padding=DF_padding_pool1,
							 name='DF_pool_1'
							 )

		# Activation pass, pooling
		a1 = (pool1(z1))
		print('DF_pool1 \t\t', a1.get_shape())

	with tf.variable_scope('DF_convLayer2'):
		### CONV2 LAYER
		print("--- Deep Fourier conv layer 2")
		### CHECK IF WEIGHT LOADING\INITIALIZE
		if ib.shall_DF_be_loaded():
			# Layer build
			z2 = tf.layers.conv1d(inputs=a1,
								  filters=DF_filters_2,
								  kernel_size=DF_kernel_size_2,
								  strides=DF_strides_conv2,
								  padding=DF_padding_conv2,
								  # data_format='channels_last',
								  # dilation_rate=1,
								  activation=tf.nn.relu,
								  use_bias=True,
								  kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[2])),
								  bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[3])),
								  # kernel_regularizer=None,
								  # bias_regularizer=None,
								  # activity_regularizer=None,
								  # kernel_constraint=None,
								  # bias_constraint=None,
								  trainable=True,
								  name="DF_conv_2",
								  # reuse=None
								  )
			print('Pretrained DF_conv1d_2 loaded!')
		else:
			# Layer build
			z2 = tf.layers.conv1d(inputs=a1,
								  filters=DF_filters_2,
								  kernel_size=DF_kernel_size_2,
								  strides=DF_strides_conv2,
								  padding=DF_padding_conv2,
								  # data_format='channels_last',
								  # dilation_rate=1,
								  activation=tf.nn.relu,
								  use_bias=True,
								  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
																						  dtype=tf.float32),
								  bias_initializer=tf.zeros_initializer(),
								  # kernel_regularizer=None,
								  # bias_regularizer=None,
								  # activity_regularizer=None,
								  # kernel_constraint=None,
								  # bias_constraint=None,
								  trainable=True,
								  name="DF_conv_2",
								  # reuse=None
								  )
			print('DF_conv1d_2 reinitialized!')

		# Input pass, activation
		print('DF_conv2 \t\t', z2.get_shape())

		### MAX_POOL2
		pool2 = MaxPooling1D(pool_size=DF_pool_size_2,
							 strides=DF_strides_pool2,
							 padding=DF_padding_pool2,
							 name='DF_pool_2'
							 )

		# Activation pass, pooling
		a2 = (pool2(z2))
		# Reshaping to swtich dimension and get them right (41x60 to 60x41x1)
		a2 = tf.transpose(a2, perm=[0, 2, 1])
		a2 = tf.expand_dims(a2, axis=3)
		# a2 = tf.reshape(a2, )
		print('DF_pool2 \t\t', a2.get_shape())

elif DF_arch == "MST":
    # Conv1
    DF_padding_conv1 = "same"
    DF_filters_1 = 512  # phi1
    DF_kernel_size_1 = 1024
    DF_strides_conv1 = 512
    
    # Conv2
    DF_padding_conv2 = 'same'
    DF_filters_2 = 256  # phi2
    DF_kernel_size_2 = 3
    DF_strides_conv2 = 1
    
    # Conv3
    DF_padding_conv3 = 'same'
    DF_filters_3 = 60  # phi3
    DF_kernel_size_3 = 3
    DF_strides_conv3 = 1
    x_pl = tf.placeholder (tf.float32, [None, length, nchannels], name='xPlaceholder_1')
    y_pl = tf.cast(tf.placeholder (tf.float64, [None, num_classes], name='yPlaceholder'), tf.float32)
    
    print('Trace of the tensors shape as it is propagated through the network.')
    print('Layer name \t Output size')
    print('----------------------------')
    ### DEEP FOURIER NETWORK
    with tf.variable_scope('DF_convLayer1'):
        ### INPUT DATA
        print("--- Deep Fourier conv layer 1")
        print('x_pl \t\t', x_pl.get_shape())
        ### CONV1 LAYER
        # Layer build
        ### CHECK IF WEIGHT LOADING\INITIALIZE
        if ib.shall_DF_be_loaded():
            z1 = tf.layers.conv1d(inputs=x_pl,
                              filters=DF_filters_1,
                              kernel_size=DF_kernel_size_1,
                              strides=DF_strides_conv1,
                              padding=DF_padding_conv1,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[0])),
                              bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[1])),
                              trainable=True,
                              name="DF_conv_1",
                              )
        else:
            z1 = tf.layers.conv1d(inputs=x_pl,
                              filters=DF_filters_1,
                              kernel_size=DF_kernel_size_1,
                              strides=DF_strides_conv1,
                              padding=DF_padding_conv1,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                                                                      dtype=tf.float32),
                              bias_initializer=tf.zeros_initializer(),
                              trainable=True,
                              name="DF_conv_1",
                              )
        # Input pass, activation
        print('DF_conv1 \t\t', z1.get_shape())
        
    with tf.variable_scope('DF_convLayer2'):
        ### CONV2 LAYER
        print("--- Deep Fourier conv layer 2")
        # Layer build
        if ib.shall_DF_be_loaded():
            z2 = tf.layers.conv1d(inputs=z1,
                              filters=DF_filters_2,
                              kernel_size=DF_kernel_size_2,
                              strides=DF_strides_conv2,
                              padding=DF_padding_conv2,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[2])),
                              bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[3])),
                              trainable=True,
                              name="DF_conv_2",
                              )
        else:
            z2 = tf.layers.conv1d(inputs=z1,
                              filters=DF_filters_2,
                              kernel_size=DF_kernel_size_2,
                              strides=DF_strides_conv2,
                              padding=DF_padding_conv2,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                                                                      dtype=tf.float32),
                              bias_initializer=tf.zeros_initializer(),
                              trainable=True,
                              name="DF_conv_2",
                              )
        # Input pass, activation
        print('DF_conv2 \t\t', z2.get_shape())
        
    with tf.variable_scope('DF_convLayer3'):
        if ib.shall_DF_be_loaded():
            z3 = tf.layers.conv1d(inputs=z2,
                              filters=DF_filters_3,
                              kernel_size=DF_kernel_size_3,
                              strides=DF_strides_conv3,
                              padding=DF_padding_conv3,
                              activation=tf.nn.tanh,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[4])),
                              bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[5])),
                              trainable=True,
                              name="DF_conv_3",
                              )
        else:
            z3 = tf.layers.conv1d(inputs=z2,
                              filters=DF_filters_3,
                              kernel_size=DF_kernel_size_3,
                              strides=DF_strides_conv3,
                              padding=DF_padding_conv3,
                              activation=tf.nn.tanh,
                              use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                                                                      dtype=tf.float32),
                              bias_initializer=tf.zeros_initializer(),
                              trainable=True,
                              name="DF_conv_3",
                              )
        # Input pass, activation
        # Reshaping to swtich dimension and get them right (41x60 to 60x41x1)
    a2 = tf.expand_dims(tf.transpose(z3, perm=[0, 2, 1]), axis=-1)
    print('Output \t\t',a2.get_shape())

elif DF_arch == "Heuri2":
    DF_padding_conv1 = "same"
    DF_filters_1 = 80  # phi1
    DF_kernel_size_1 = 1024
    DF_strides_conv1 = 512

    DF_padding_conv2 = 'same'
    DF_filters_2 = 60  # phi2
    DF_kernel_size_2 = 15
    DF_strides_conv2 = 1
    x_pl = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')
    y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
    y_pl = tf.cast(y_pl, tf.float32)
    with tf.variable_scope('DF_convLayer1'):
        if ib.shall_DF_be_loaded():
            z1 = tf.layers.conv1d(inputs=x_pl,
                              filters=DF_filters_1,
                              kernel_size=DF_kernel_size_1,
                              strides=DF_strides_conv1,
                              padding=DF_padding_conv1,
                              activation=tf.abs,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[0])),
                              bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[1])),
                              trainable=True,
                              name="DF_conv_1",
                              )
        else:
            z1 = tf.layers.conv1d(inputs=x_pl,
                              filters=DF_filters_1,
                              kernel_size=DF_kernel_size_1,
                              strides=DF_strides_conv1,
                              padding=DF_padding_conv1,
                              activation=tf.abs,
                              use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                                                                      dtype=tf.float32),
                              bias_initializer=tf.zeros_initializer(),
                              trainable=True,
                              name="DF_conv_1",
                              )
    
    with tf.variable_scope('DF_convLayer2'):
        if ib.shall_DF_be_loaded():
            a2 = tf.layers.conv1d(inputs=z1,
                              filters=DF_filters_2,
                              kernel_size=DF_kernel_size_2,
                              strides=DF_strides_conv2,
                              padding=DF_padding_conv2,
                              activation=tf.nn.tanh,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[2])),
                              bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_DF[3])),
                              trainable=True,
                              name="DF_conv_2",
                              )
        else:
            a2 = tf.layers.conv1d(inputs=z1,
                              filters=DF_filters_2,
                              kernel_size=DF_kernel_size_2,
                              strides=DF_strides_conv2,
                              padding=DF_padding_conv2,
                              activation=tf.nn.tanh,
                              use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                                                                      dtype=tf.float32),
                              bias_initializer=tf.zeros_initializer(),
                              trainable=True,
                              name="DF_conv_2",
                              )
        a2 = tf.expand_dims(tf.transpose(a2,perm=[0,2,1]), axis=3)
                              
                              
                              
elif DF_arch == "MelNet": 
    raise Exception("MelNet not yet implemented!")
    x_pl = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')
    y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
    y_pl = tf.cast(y_pl, tf.float32)

else: 
    raise Exception("Specify a DF architecture!")

### PICZAK NETWORK
# Convolutional layers
from dimensions_PZ import *

with tf.variable_scope('PZ_convLayer1'):    
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if True:
        print("--- Piczak")
        conv1 = tf.layers.conv2d(
            kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[0])),
            bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[1])),
            inputs=a2,  ### NB!!! This is the output of the Deep_Fourier Network
            filters=n_filter_1,
            kernel_size=kernel_size_1,
            strides=kernel_strides_1,
            padding=padding_1,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_1))
        print('Pretrained PZ_conv2d_1 loaded!')
    else:
        raise Exception("this does not get executed!")

    x = conv1
    print('PZ_conv1 \t\t', x.get_shape())
    pool1 = max_pool2d(x, kernel_size=pool_size_1, stride=pool_strides_1, padding=padding_1)
    x = pool1
    print('PZ_pool1 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_1)

with tf.variable_scope('PZ_convLayer2'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if True:
        conv2 = tf.layers.conv2d(
            kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[2])),
            bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[3])),
            inputs=x,
            filters=n_filter_2,
            kernel_size=kernel_size_2,
            strides=kernel_strides_2,
            padding=padding_2,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_2))
        print('Pretrained PZ_conv2d_2 loaded!')
    else:
        raise Exception("this does not get executed!")

    x = conv2
    print('PZ_conv2 \t\t', x.get_shape())
    pool2 = max_pool2d(x, kernel_size=pool_size_2, stride=pool_strides_2, padding=padding_2)
    x = pool2
    print('PZ_pool2 \t\t', x.get_shape())
    # We flatten x for dense layers
    x = flatten(x)
    print('PZ_Flatten \t', x.get_shape())

# Dense layers
with tf.variable_scope('PZ_denseLayer3'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if True:
        dense3 = tf.layers.dense(
            kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[4])),
            bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[5])),
            inputs=x,
            units=num_units_3,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_3),
        )
        print('Pretrained PZ_dense_1 loaded!')
    else:
        raise Exception("this does not get executed!")

    x = dense3
    print('PZ_dense3 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_3)

with tf.variable_scope('PZ_denseLayer4'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if True:
        dense4 = tf.layers.dense(
            kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[6])),
            bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[7])),
            inputs=x,
            units=num_units_4,
            activation=tf.nn.relu,
            # kernel_initializer=None,
            # bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_4),
        )
        print('Pretrained PZ_dense_2 loaded!')
    else:
        raise Exception("this does not get executed!")

    x = dense4
    print('PZ_dense4 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_4)

with tf.variable_scope('PZ_output_layer'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if True:
        dense_out = tf.layers.dense(
            kernel_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[8])),
            bias_initializer=tf.constant_initializer(np.copy(ib.pretrained_PZ[9])),
            inputs=x,
            units=num_classes,
            activation=None,
            # kernel_initializer=None,
            # bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_output),
        )
        print('Pretrained PZ_output loaded!')
    else:
        raise Exception("this does not get executed!")

    y = tf.nn.softmax(dense_out)
    print('denseOut\t', y.get_shape())

print('Model consists of ', utils.num_params(), 'trainable parameters.')
# %%
#########################################
###   SETTING VARIABLES TRAINABILITY  ###
#########################################

### STORING TRAINABLE VARIABLES
all_vars = tf.trainable_variables()
start_idx, end_idx = ib.what_is_trainable(all_vars)
to_train = all_vars[start_idx:end_idx]

print("and we will train: ")
for j in range(len(to_train)):
    print("## ", to_train[j])

####################################################
###   LOSS, TRAINING AND PERFORMANCE DEFINITION  ###
####################################################

with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y + 1e-8), reduction_indices=[1])
    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)
    if STORE_GRADIENT_NORMS: calculate_gradient_norms_l2 = [ tf.norm(tf.gradients(cross_entropy, all_vars[j]), 2, name="gradnorm_l2_%d"%j) for j in range(len(all_vars)) ]

with tf.variable_scope('training'):
	LR = tf.placeholder(tf.float64, shape=[]) #that way we can change the learning rate on the fly
	sgd = tf.train.MomentumOptimizer(learning_rate=LR, momentum=momentum, use_nesterov=True)
	grads_and_vars = sgd.compute_gradients(cross_entropy, var_list=to_train)
	if USE_GRADIENT_CLIPPING:	
		clipped_gradients = [(tf.clip_by_norm(grad, 3), var) for grad, var in grads_and_vars]
		train_op = sgd.apply_gradients(clipped_gradients)
	else:
		train_op = sgd.apply_gradients(grads_and_vars)

with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# %%
##############################
###   FORWARD PASS TESTING ###
##############################

# Random audio images for testing
x_test_forward = np.random.normal(0, 1, [50, 20992, 1]).astype('float32')  # dummy data
y_dummy_train = utils.onehot(np.random.randint(0, 10, 50), 10)

if RUN_FAST: gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)# don't kill the laptop
else: gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
sess.run(tf.global_variables_initializer())
eat_this = {x_pl: x_test_forward, y_pl: y_dummy_train}
res_forward_pass = sess.run(fetches=[y], feed_dict=eat_this)

print("y", res_forward_pass[0].shape)
print('Forward pass successful!')

# %%
##########################
###   FOLDS LOADER     ###
##########################
# Folds creation : lists with all the np.array's inside
data_folds = []
labels_folds = []
if RUN_FAST: small_data = batch_size // 10 + 5
for i in range(1, 11):
    if RUN_FAST: 
        data_mat = np.random.normal(0, 1, [small_data, 20992, 1]).astype('float32')
    else: 
        data_mat = scipy.io.loadmat(data_folder + 'fold{}_wav.mat'.format(i))
        # Add one dimension for being eligible for the network
        data_mat = np.expand_dims(data_mat['ob_wav'], axis=-1)
    data_folds.append(data_mat)
    del data_mat #unburden the memory
    if RUN_FAST: 
        labels_mat = utils.onehot(np.random.randint(0, 10, small_data), 10)
    else: 
        labels_mat = scipy.io.loadmat(data_folder + 'fold{}_labels.mat'.format(i))
        labels_mat = utils.onehot(np.transpose(labels_mat['lb']), num_classes)  # One-hot encoding labels
    labels_folds.append(labels_mat)
    del labels_mat #unburden the memory

# %%
##########################
###   TRAINING LOOP    ###
##########################

with tf.Session() as sess:
    # Cross-validation
    try:
        print("------------------------------------------------------------")
        print('----A method : training on all folds but no. {0} (validation) and {1}(test)'.format(k_valid + 1,
                                                                                                   k_test + 1))
        print("------------------------------------------------------------")

        # We reinitialize the weights
        sess.run(tf.global_variables_initializer())
        epoch = iteration_ctr = 0

        mask = [True] * n_fold
        for k in [k_valid, k_test]:
            mask[k] = False
        train_data = [data_folds[i] for i in range(len(mask)) if mask[i]]
        train_labels = [labels_folds[i] for i in range(len(mask)) if mask[i]]
        # Merging data (list being different from np.arrays)
        merged_train_data = np.empty((0, length, n_channels))
        merged_train_labels = np.empty((0, num_classes))
        for i_merge in range(n_fold - 2):
            merged_train_data = np.vstack((merged_train_data, train_data[i_merge]))
            merged_train_labels = np.vstack((merged_train_labels, train_labels[i_merge]))

        train_data = merged_train_data
        train_labels = merged_train_labels
        del merged_train_data, merged_train_labels

        train_loader = bl.batch_loader(train_data, train_labels, batch_size, is_balanced=BALANCED_BATCHES,
                                       is_fast=RUN_FAST)

        valid_data = data_folds[k_valid]
        valid_labels = labels_folds[k_valid]

        # Training loss and accuracy for each epoch : initialization
        train_loss, train_accuracy = [], []
        # Training loss and accuracy within an epoch (is erased at every new epoch)
        _train_loss, _train_accuracy = [], []
        if STORE_GRADIENT_NORMS: _gradientnorms_l2 = np.empty((0, len(all_vars)), float)
        valid_loss, valid_accuracy = [], []
        bal_valid_accuracy = []
        test_loss, test_accuracy = [], []

        ### TRAINING ###
        TIME_epoch_start = time.time()
        while (epoch < max_epochs):
            train_batch_data, train_batch_labels = train_loader.next_batch()
            feed_dict_train = {x_pl: train_batch_data, y_pl: train_batch_labels, LR: lr_array[epoch]}
            # deciding which parts to fetch, train_op makes the classifier "train"
            fetches_train = [train_op, cross_entropy, accuracy]
            # running the train_op and computing the updated training loss and accuracy
            if STORE_GRADIENT_NORMS and iteration_ctr % nskip_iter_GN == 0:
                res = sess.run(fetches=[calculate_gradient_norms_l2, train_op, cross_entropy, accuracy], feed_dict=feed_dict_train)
                # storing cross entropy (second fetch argument, so index=1)
                _train_loss += [res[2]]
                _train_accuracy += [res[3]]
                _gradientnorms_l2 = np.vstack((_gradientnorms_l2, np.array(res[0])))
            else:
                res = sess.run(fetches=[train_op, cross_entropy, accuracy], feed_dict=feed_dict_train)
                # storing cross entropy (second fetch argument, so index=1)
                _train_loss += [res[1]]
                _train_accuracy += [res[2]]
            iteration_ctr+=1
            ### VALIDATING ###
            # When we reach the last mini-batch of the epoch
            if train_loader.is_epoch_done():
                # what to feed our accuracy op
                feed_dict_valid = {x_pl: valid_data, y_pl: valid_labels}
                # deciding which parts to fetch
                fetches_valid = [cross_entropy, accuracy]
                # running the validation
                res = sess.run(fetches=fetches_valid, feed_dict=feed_dict_valid)
                # Update all accuracies
                valid_loss += [res[0]]
                valid_accuracy += [res[1]]
                train_loss += [np.mean(_train_loss)]
                train_accuracy += [np.mean(_train_accuracy)]
                # Balanced validation accuracy
                pred_labels = np.argmax(sess.run(fetches=y, feed_dict={x_pl: valid_data}), axis=1)
                true_labels = utils.onehot_inverse(valid_labels)
                conf_mat = confusion_matrix(true_labels, pred_labels, labels=range(10))
                bal_valid_accuracy += [utils.classbal_acc(conf_mat)]
                # Reinitialize the intermediate loss and accuracy within epochs
                _train_loss, _train_accuracy = [], []
                # Print a summary of the training and validation
                print(
                    "Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}, took {:10.2f} sec".format(
                        epoch, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1],
                        time.time() - TIME_epoch_start))
                print("")
                TIME_epoch_start = time.time()
                # "Early stopping" (in fact, we keep going but just take the best network at every time step we have improvement)
                if valid_accuracy[-1] == max(valid_accuracy):
                    # Updating the best quantities
                    best_train_loss = train_loss[-1]
                    best_train_accuracy = train_accuracy[-1]
                    best_valid_loss = valid_loss[-1]
                    best_valid_accuracy = valid_accuracy[-1]
                    best_epoch = epoch

                    # Weights
                    variables_names = [v.name for v in tf.trainable_variables()]
                    best_weights = sess.run(variables_names)

                if bal_valid_accuracy[-1] == max(bal_valid_accuracy):
                    # Updating the best quantities
                    best_bal_train_loss = train_loss[-1]
                    best_bal_train_accuracy = train_accuracy[-1]
                    best_bal_valid_loss = valid_loss[-1]
                    best_bal_valid_accuracy = bal_valid_accuracy[-1]
                    best_bal_epoch = epoch
                    # Weights
                    variables_names = [v.name for v in tf.trainable_variables()]
                    best_bal_weights = sess.run(variables_names)
                # Update epoch
                epoch += 1;
        # Save everything (all training history + the best values)
        mdict = {'train_loss': train_loss, 'train_accuracy': train_accuracy, 'valid_loss': valid_loss,
                 'valid_accuracy': valid_accuracy, 'best_train_loss': best_train_loss,
                 'best_train_accuracy': best_train_accuracy, 'best_valid_loss': best_valid_loss,
                 'best_valid_accuracy': best_valid_accuracy, 'best_epoch': best_epoch,
                 'bal_valid_accuracy': bal_valid_accuracy, 'best_bal_train_loss': best_bal_train_loss,
                 'best_bal_train_accuracy': best_bal_train_accuracy, 'best_bal_valid_loss': best_bal_valid_loss,
                 'best_bal_valid_accuracy': best_bal_valid_accuracy, 'best_bal_epoch': best_bal_epoch}
        scipy.io.savemat(ib.good_place_to_store_perf(), mdict)
        if STORE_GRADIENT_NORMS: scipy.io.savemat(save_path_perf + filename + "_GRADIENTNORMS", {'GN_L2': _gradientnorms_l2})
        ib.save_weights(best_bal_weights)
    except KeyboardInterrupt:
        pass

print("<><><><><><><><> the entire program finished without errors!! <><><><><><><><>")

