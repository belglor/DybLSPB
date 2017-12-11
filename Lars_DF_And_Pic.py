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

###SET RANDOM SEED AND RESET GRAPH
tf.reset_default_graph()

# =============================================================================
### SETTING INITIALIZATION FLAGS

LOAD_PRETRAINED_DF = False #this is temporary, because the loaded DF mat file does not have all the needed entries yet

# Select which weights to load for DF!
if LOAD_PRETRAINED_DF:
    DF_toload_weights = "results_mat/trainedweights/deepFourier_dfT_dfcnn1X_dfcnn2X_pzcnn1TX_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-002_ME300_WEIGHTS.mat"
    print('LOADING PRETRAINED DF: ' + DF_toload_weights)
    tw_DF = scipy.io.loadmat(DF_toload_weights);
# Select which layers to load
DF_load_conv1d_1 = False;
DF_load_conv1d_2 = False;
DF_load_conv1d_3 = False;

# Select which weights to load for PZ!
PZ_toload_weights = "results_mat/trainedweights/piczak_A_unbal_LR0-002_ME300_WEIGHTS"
print('LOADING PRETRAINED PZ: ' + PZ_toload_weights)
tw_PZ = scipy.io.loadmat(PZ_toload_weights);
# Select which layers to load
PZ_load_conv2d_1 = True;
PZ_load_conv2d_2 = True;
PZ_load_dense_1 = True;
PZ_load_dense_2 = True;
PZ_load_output = True;

# =============================================================================

# %%
#..................................
###################################
###   CODE SETTINGS   #############
###################################
### SETTING TRAINABILITY FLAGS
DF_trainable = True
PZ_1stCNN_trainable = True  # True for Lars1, False for Ole1
PZ_2ndCNN_trainable = False
PZ_FullyC_trainable = False

n_fold = 10
# Folds we validate and test on (only relevant if RUN_CV==False). Indices of the folds are in the "Matlab" naming mode
k_valid = 9
k_test = 10
# If we just want to test quickly on a few epochs
RUN_FAST = True
# If we want oversampled (balanced) mini-batches or not
BALANCED_BATCHES = False
# Learning rate
learning_rate = 0.005
# Number of epochs (only one is relevant acc. to which RUN_FAST value has been given)
max_epochs_fast = 2
max_epochs_regular = 300
# Batch size
batch_size = 1000
USE_GRADIENT_CLIPPING = True
STORE_GRADIENT_NORMS = True
USE_LR_DECAY = False #learning rate decay: exponentially decreasing the learning rate over the epochs, so that we do not overshoot the (local) optimum when we get close
learning_rate_END = .0001 #if USE_LR_DECAY is set to True, then the learning rate will exponentially decay at every epoch and will reach learning_rate_END at the last epoch
NAME_SAFETY_PREFIX = "" #set this to "" for normal behaviour. Set it to any other string if you are just testing things and want to avvoid overwriting important stuff
nskip_iter_GN = 20 #every nskip_iter_GN'th iteration we will take one measurement of the gradient L2 norm (measuring it at every iteration takes too much comp. time)  
###################################
###  END OF CODE SETTINGS   #######
###################################
#..................................

# =============================================================================

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

# create these 3 folders if you don't have them
# if you really have to change these folders, do it HERE and not further down in the code, and do not git push these folders
data_folder = "./data/"
result_mat_folder = "./results_mat/"

save_path_perf = result_mat_folder + "performance/"
save_path_numpy_weights = result_mat_folder + "trainedweights/"

for directory in [save_path_perf, save_path_numpy_weights]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Filename definition
word_cv = 'A'

# Trainability labels
if DF_trainable:
    word_df = 'T'
else:
    word_df = 'N'
if PZ_1stCNN_trainable:
    word_cnn1 = 'T'
else:
    word_cnn1 = 'N'
if PZ_2ndCNN_trainable:
    word_cnn2 = 'T'
else:
    word_cnn2 = 'N'
if PZ_FullyC_trainable:
    word_fc = 'T'
else:
    word_fc = 'N'

# Weight loading\initialization labels
if DF_load_conv1d_1:
    word_dfcnn1 = "L"
else:
    word_dfcnn1 = "X"
if DF_load_conv1d_2:
    word_dfcnn2 = "L"
else:
    word_dfcnn2 = "X"
if DF_load_conv1d_3:
    word_dfcnn3 = "L"
else:
    word_dfcnn3 = "X"


if PZ_load_conv2d_1:
    word_pzcnn1 = "L"
else:
    word_pzcnn1 = "X"
if PZ_load_conv2d_2:
    word_pzcnn2 = "L"
else:
    word_pzcnn2 = "X"
if PZ_load_dense_1:
    word_pzfc1 = "L"
else:
    word_pzfc1 = "X"
if PZ_load_dense_2:
    word_pzfc2 = "L"
else:
    word_pzfc2 = "X"
if PZ_load_output:
    word_pzout = "L"
else:
    word_pzout = "X"

if BALANCED_BATCHES:
    word_bal = 'bal'
else:
    word_bal = 'unbal'
    
if USE_LR_DECAY: word_lrd = "_LRD"
else: word_lrd = ""

word_lr = str(learning_rate)
word_lr = word_lr[:1] + '-' + word_lr[2:]

# Filename typesetting:
# deepFourier_df{trainable}_dfcnn1{init}_dfcnn2{init}_pzcnn1{trainable}{init}_pzcnn2{trainable}{init}_pzfc1{trainable}{init}_pzfc2{trainable}{init}_pzout{init}_{word_cv}_{word_bal}_LR{word_lr}_ME{max_epochs}
filename = NAME_SAFETY_PREFIX + "LarsDeepFourier_df{0}_dfcnn1{1}_dfcnn2{2}_dfcnn3{3}_pzcnn1{4}{5}_pzcnn2{6}{7}_pzfc1{8}{9}_pzfc2{10}{11}_pzout{12}_{13}_{14}_LR{15}_ME{16}{17}".format(
    word_df, word_dfcnn1, word_dfcnn2, word_dfcnn3, word_cnn1, word_pzcnn1, word_cnn2, word_pzcnn2, word_fc, word_pzfc1, word_fc,
    word_pzfc2, word_pzout, word_cv, word_bal, word_lr, max_epochs, word_lrd)
# %%
################################
###   GET  TRAINED WEIGHTS   ###
################################
# Separate weights for DF
if LOAD_PRETRAINED_DF:
    pretrained_conv1d_1_kernel_DF = tw_DF['DF_conv1d_1_kernel']
    pretrained_conv1d_1_bias_DF = tw_DF['DF_conv1d_1_bias']
    pretrained_conv1d_2_kernel_DF = tw_DF['DF_conv1d_2_kernel']
    pretrained_conv1d_2_bias_DF = tw_DF['DF_conv1d_2_bias']
    pretrained_conv1d_3_kernel_DF = tw_DF['DF_conv1d_3_kernel']
    pretrained_conv1d_3_bias_DF = tw_DF['DF_conv1d_3_bias']

# Separate weights for PZ
pretrained_conv2d_1_kernel_PZ = tw_PZ['conv2d_1_kernel']
pretrained_conv2d_1_bias_PZ = tw_PZ['conv2d_1_bias']
pretrained_conv2d_2_kernel_PZ = tw_PZ['conv2d_2_kernel']
pretrained_conv2d_2_bias_PZ = tw_PZ['conv2d_2_bias']
pretrained_dense_1_kernel_PZ = tw_PZ['dense_1_kernel']
pretrained_dense_1_bias_PZ = tw_PZ['dense_1_bias']
pretrained_dense_2_kernel_PZ = tw_PZ['dense_2_kernel']
pretrained_dense_2_bias_PZ = tw_PZ['dense_2_bias']
pretrained_output_kernel_PZ = tw_PZ['output_kernel']
pretrained_output_bias_PZ = tw_PZ['output_bias']

# %%
##############################
###   NETWORK PARAMETERS   ###
##############################

###GENERAL VARIABLES
# Input data size
length = 20992  # k0
nchannels = 1  # ??? Maybe two because stereo audiofiles?

###DEEP_FOURIER LAYERS HYPERPARAMETERS
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

###PICZACK HYPERPARAMETERS
# Bands : related to frequencies. Frames : related to audio sequence. 2 channels (the spec and the delta's)
bands, frames, n_channels = 60, 41, 1
image_shape = [bands, frames, n_channels]

# First convolutional ReLU layer
n_filter_1 = 80
kernel_size_1 = [57, 6]
kernel_strides_1 = (1, 1)
# Activation in the layer definition
# activation_1="relu"
# L2 weight decay
l2_1 = 0.001

# Dropout rate after pooling
dropout_1 = 0.5

# First MaxPool layer
pool_size_1 = (4, 3)
pool_strides_1 = (1, 3)
padding_1 = "valid"

### Second convolutional ReLU layer
n_filter_2 = 80
kernel_size_2 = [1, 3]
kernel_strides_2 = (1, 1)
# Activation in the layer definition
# activation_2="relu"
l2_2 = 0.001

# Scond MaxPool layer
pool_size_2 = (1, 3)
pool_strides_2 = (1, 3)
padding_2 = "valid"

# Third (dense) ReLU layer
num_units_3 = 5000
# Activation in the layer definition
# activation_3 = "relu"
dropout_3 = 0.5
l2_3 = 0.001

# Fourth (dense) ReLU layer
num_units_4 = 5000
# Activation in the layer definition
# activation_4 = "relu"
dropout_4 = 0.5
l2_4 = 0.001

# Output softmax layer (10 classes in UrbanSound8K)
num_classes = 10
# Activation in the layer definition
# activation_output="softmax"
l2_output = 0.001

# Learning rate
momentum = 0.9

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
    if DF_load_conv1d_1:
        z1 = tf.layers.conv1d(inputs=x_pl,
                              filters=DF_filters_1,
                              kernel_size=DF_kernel_size_1,
                              strides=DF_strides_conv1,
                              padding=DF_padding_conv1,
                              # data_format='channels_last',
                              # dilation_rate=1,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(pretrained_conv1d_1_kernel_DF),
                              bias_initializer=tf.constant_initializer(pretrained_conv1d_1_bias_DF),
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

with tf.variable_scope('DF_convLayer2'):
    ### CONV2 LAYER
    print("--- Deep Fourier conv layer 2")
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if DF_load_conv1d_2:
        # Layer build
        z2 = tf.layers.conv1d(inputs=z1,
                              filters=DF_filters_2,
                              kernel_size=DF_kernel_size_2,
                              strides=DF_strides_conv2,
                              padding=DF_padding_conv2,
                              # data_format='channels_last',
                              # dilation_rate=1,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(pretrained_conv1d_2_kernel_DF),
                              bias_initializer=tf.constant_initializer(pretrained_conv1d_2_bias_DF),
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
        z2 = tf.layers.conv1d(inputs=z1,
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

with tf.variable_scope('DF_convLayer3'):
    ### CONV2 LAYER
    print("--- Deep Fourier conv layer 3")
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if DF_load_conv1d_3:
        # Layer build
        z3 = tf.layers.conv1d(inputs=z2,
                              filters=DF_filters_3,
                              kernel_size=DF_kernel_size_3,
                              strides=DF_strides_conv3,
                              padding=DF_padding_conv3,
                              # data_format='channels_last',
                              # dilation_rate=1,
                              activation=tf.nn.tanh,
                              use_bias=True,
                              kernel_initializer=tf.constant_initializer(pretrained_conv1d_3_kernel_DF),
                              bias_initializer=tf.constant_initializer(pretrained_conv1d_3_bias_DF),
                              # kernel_regularizer=None,
                              # bias_regularizer=None,
                              # activity_regularizer=None,
                              # kernel_constraint=None,
                              # bias_constraint=None,
                              trainable=True,
                              name="DF_conv_3",
                              # reuse=None
                              )
        print('Pretrained DF_conv1d_3 loaded!')
    else:
        # Layer build
        z3 = tf.layers.conv1d(inputs=z2,
                              filters=DF_filters_3,
                              kernel_size=DF_kernel_size_3,
                              strides=DF_strides_conv3,
                              padding=DF_padding_conv3,
                              # data_format='channels_last',
                              # dilation_rate=1,
                              activation=tf.nn.tanh,
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
                              name="DF_conv_3",
                              # reuse=None
                              )
        print('DF_conv1d_3 reinitialized!')

    # Input pass, activation
    # Reshaping to swtich dimension and get them right (to 41x60 to 60x41x1)
    z3 = tf.transpose(z3, perm=[0, 2, 1])
    z3 = tf.expand_dims(z3, axis=3)

    print('DF_conv3 \t\t', z3.get_shape())

### PICZAK NETWORK
# Convolutional layers
with tf.variable_scope('PZ_convLayer1'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if PZ_load_conv2d_1:
        print("--- Piczak")
        conv1 = tf.layers.conv2d(
            kernel_initializer=tf.constant_initializer(pretrained_conv2d_1_kernel_PZ),
            bias_initializer=tf.constant_initializer(pretrained_conv2d_1_bias_PZ),
            inputs=z3,  ### NB!!! This is the output of the Deep_Fourier Network
            filters=n_filter_1,
            kernel_size=kernel_size_1,
            strides=kernel_strides_1,
            padding=padding_1,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_1))
        print('Pretrained PZ_conv2d_1 loaded!')
    else:
        print("--- Piczak")
        conv1 = tf.layers.conv2d(
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
            bias_initializer=tf.zeros_initializer(),
            inputs=z3,  ### NB!!! This is the output of the Deep_Fourier Network
            filters=n_filter_1,
            kernel_size=kernel_size_1,
            strides=kernel_strides_1,
            padding=padding_1,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_1))
        print('PZ_conv2d_1 reinitialized!')

    x = conv1
    print('PZ_conv1 \t\t', x.get_shape())
    pool1 = max_pool2d(x, kernel_size=pool_size_1, stride=pool_strides_1, padding=padding_1)
    x = pool1
    print('PZ_pool1 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_1)

with tf.variable_scope('PZ_convLayer2'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if PZ_load_conv2d_2:
        conv2 = tf.layers.conv2d(
            kernel_initializer=tf.constant_initializer(pretrained_conv2d_2_kernel_PZ),
            bias_initializer=tf.constant_initializer(pretrained_conv2d_2_bias_PZ),
            inputs=x,
            filters=n_filter_2,
            kernel_size=kernel_size_2,
            strides=kernel_strides_2,
            padding=padding_2,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_2))
        print('Pretrained PZ_conv2d_2 loaded!')
    else:
        conv2 = tf.layers.conv2d(
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
            bias_initializer=tf.zeros_initializer(),
            inputs=x,
            filters=n_filter_2,
            kernel_size=kernel_size_2,
            strides=kernel_strides_2,
            padding=padding_2,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_2))
        print('PZ_conv2d_2 reinitialized!')

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
    if PZ_load_dense_1:
        dense3 = tf.layers.dense(
            kernel_initializer=tf.constant_initializer(pretrained_dense_1_kernel_PZ),
            bias_initializer=tf.constant_initializer(pretrained_dense_1_bias_PZ),
            inputs=x,
            units=num_units_3,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_3),
        )
        print('Pretrained PZ_dense_1 loaded!')
    else:
        dense3 = tf.layers.dense(
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
            bias_initializer=tf.zeros_initializer(),
            inputs=x,
            units=num_units_3,
            activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_3),
        )
        print('PZ_dense_1 reinitialized!')

    x = dense3
    print('PZ_dense3 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_3)

with tf.variable_scope('PZ_denseLayer4'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if PZ_load_dense_2:
        dense4 = tf.layers.dense(
            kernel_initializer=tf.constant_initializer(pretrained_dense_2_kernel_PZ),
            bias_initializer=tf.constant_initializer(pretrained_dense_2_bias_PZ),
            inputs=x,
            units=num_units_4,
            activation=tf.nn.relu,
            # kernel_initializer=None,
            # bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_4),
        )
        print('Pretrained PZ_dense_2 loaded!')
    else:
        dense4 = tf.layers.dense(
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
            bias_initializer=tf.zeros_initializer(),
            inputs=x,
            units=num_units_4,
            activation=tf.nn.relu,
            # kernel_initializer=None,
            # bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_4),
        )
        print('PZ_dense_2 reinitialized!')

    x = dense4
    print('PZ_dense4 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_4)

with tf.variable_scope('PZ_output_layer'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    if PZ_load_output:
        dense_out = tf.layers.dense(
            kernel_initializer=tf.constant_initializer(pretrained_output_kernel_PZ),
            bias_initializer=tf.constant_initializer(pretrained_output_bias_PZ),
            inputs=x,
            units=num_classes,
            activation=None,
            # kernel_initializer=None,
            # bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_output),
        )
        print('Pretrained PZ_output loaded!')
    else:
        dense_out = tf.layers.dense(
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
            bias_initializer=tf.zeros_initializer(),
            inputs=x,
            units=num_classes,
            activation=None,
            # kernel_initializer=None,
            # bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_output),
        )
        print('PZ_output reinitialized!')

    y = tf.nn.softmax(dense_out)
    print('denseOut\t', y.get_shape())

print('Model consists of ', utils.num_params(), 'trainable parameters.')
# %%
#########################################
###   SETTING VARIABLES TRAINABILITY  ###
#########################################

### STORING TRAINABLE VARIABLES
all_vars = tf.trainable_variables()

### SLICING VARIABLES
# Deep Fourier training variables
DF_trainable_stuff = all_vars[0:6]

# Piczak (1st CNN) training variables
PZ_1stCNN_trainable_stuff = all_vars[6:8]

# Piczak (2nd CNN) training variables
PZ_2ndCNN_trainable_stuff = all_vars[8:10]

# Piczak (Fully Connected) training variables
PZ_FullyC_trainable_stuff = all_vars[10:12]

### CREATING LIST OF VARIABLES TO TRAIN
to_train = list()
if (DF_trainable): to_train += DF_trainable_stuff
if (PZ_1stCNN_trainable): to_train += PZ_1stCNN_trainable_stuff
if (PZ_2ndCNN_trainable): to_train += PZ_2ndCNN_trainable_stuff
if (PZ_FullyC_trainable): to_train += PZ_FullyC_trainable_stuff

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
		print("grads_and_vars has length: ", len(grads_and_vars))
		print("....this is grads_and_vars[0]", grads_and_vars[0])
		print("....this is grads_and_vars[0][0]", grads_and_vars[0][0])
		print("....this is grads_and_vars[0][1]", grads_and_vars[0][1])
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

# This hell line
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

        # ### COMMENT THIS OUT AND UNCOMMENT ABOVE FOR FULL TRAINING DATA
        #        i_merge = 1;
        #        merged_train_data = np.vstack((merged_train_data, train_data[i_merge]))
        #        merged_train_labels = np.vstack((merged_train_labels, train_labels[i_merge]))

        train_data = merged_train_data
        train_labels = merged_train_labels
        del merged_train_data, merged_train_labels

        train_loader = bl.batch_loader(train_data, train_labels, batch_size, is_balanced=BALANCED_BATCHES, is_fast=RUN_FAST)

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
                print("Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}, took {:10.2f} sec".format(
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
                epoch += 1;
        # Save everything (all training history + the best values)
        mdict = {'train_loss': train_loss, 'train_accuracy': train_accuracy, 'valid_loss': valid_loss,
                 'valid_accuracy': valid_accuracy, 'best_train_loss': best_train_loss,
                 'best_train_accuracy': best_train_accuracy, 'best_valid_loss': best_valid_loss,
                 'best_valid_accuracy': best_valid_accuracy, 'best_epoch': best_epoch,
                 'bal_valid_accuracy': bal_valid_accuracy, 'best_bal_train_loss': best_bal_train_loss,
                 'best_bal_train_accuracy': best_bal_train_accuracy, 'best_bal_valid_loss': best_bal_valid_loss,
                 'best_bal_valid_accuracy': best_bal_valid_accuracy, 'best_bal_epoch': best_bal_epoch}
        scipy.io.savemat(save_path_perf + filename + "_ACCURACY", mdict)
        if STORE_GRADIENT_NORMS: scipy.io.savemat(save_path_perf + filename + "_GRADIENTNORMS", {'GN_L2': _gradientnorms_l2})

        # Saving the weights
        TIME_saving_start = time.time()
        print("performance saved under %s...." % save_path_perf)
        scipy.io.savemat(save_path_numpy_weights + filename + "_WEIGHTS", dict(
            DF_conv1d_1_kernel=best_weights[0],
            DF_conv1d_1_bias=best_weights[1],
            DF_conv1d_2_kernel=best_weights[2],
            DF_conv1d_2_bias=best_weights[3],
            DF_conv1d_3_kernel=best_weights[4],
            DF_conv1d_3_bias=best_weights[5],
            PZ_conv2d_1_kernel=best_weights[6],
            PZ_conv2d_1_bias=best_weights[7],
            PZ_conv2d_2_kernel=best_weights[8],
            PZ_conv2d_2_bias=best_weights[9],
            dense_1_kernel=best_weights[10],
            dense_1_bias=best_weights[11],
            dense_2_kernel=best_weights[12],
            dense_2_bias=best_weights[13],
            output_kernel=best_weights[14],
            output_bias=best_weights[15]
        ))
        print("weights (numpy arrays) saved under %s....: " % save_path_numpy_weights)
        print("... saving took {:10.2f} sec".format(time.time() - TIME_saving_start))

        # Saving the weights
        # TIME_saving_start = time.time()
        scipy.io.savemat(save_path_numpy_weights + filename + "_BAL_WEIGHTS", dict(
            DF_conv1d_1_kernel=best_bal_weights[0],
            DF_conv1d_1_bias=best_bal_weights[1],
            DF_conv1d_2_kernel=best_bal_weights[2],
            DF_conv1d_2_bias=best_bal_weights[3],
            DF_conv1d_3_kernel=best_bal_weights[4],
            DF_conv1d_3_bias=best_bal_weights[5],
            PZ_conv2d_1_kernel=best_bal_weights[6],
            PZ_conv2d_1_bias=best_bal_weights[7],
            PZ_conv2d_2_kernel=best_bal_weights[8],
            PZ_conv2d_2_bias=best_bal_weights[9],
            dense_1_kernel=best_bal_weights[10],
            dense_1_bias=best_bal_weights[11],
            dense_2_kernel=best_bal_weights[12],
            dense_2_bias=best_bal_weights[13],
            output_kernel=best_bal_weights[14],
            output_bias=best_bal_weights[15]
        ))
        print("'balanced' weights (numpy arrays) saved under %s....: " % save_path_numpy_weights)
        print("... saving took {:10.2f} sec".format(time.time() - TIME_saving_start))

    except KeyboardInterrupt:
        pass

print("<><><><><><><><> the entire program finished without errors!! <><><><><><><><>")
