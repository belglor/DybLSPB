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
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn

###SET RANDOM SEED AND RESET GRAPH
tf.reset_default_graph()

##############################
###   NETWORK PARAMETERS   ###
##############################

###GENERAL VARIABLES
# Input data size
length = 20992; #k0 
nchannels = 1; #??? Maybe two because stereo audiofiles?

###DEEP_FOURIER LAYERS HYPERPARAMETERS
# Conv1, MaxPool1 parameters
DF_padding_conv1 = "valid"
DF_filters_1 = 80 #phi1
DF_kernel_size_1 = 155
DF_strides_conv1 = 10
DF_padding_pool1 = "valid"
DF_pool_size_1 = 9
DF_strides_pool1 = 3

# Conv2, MaxPool2 parameters
DF_padding_conv2 = 'valid'
DF_filters_2 = 60 #phi2
DF_kernel_size_2 = 15
DF_strides_conv2 = 4
DF_padding_pool2 = 'valid'
DF_pool_size_2 = 9
DF_strides_pool2 = 4

###PICZACK HYPERPARAMETERS
# Bands : related to frequencies. Frames : related to audio sequence. 2 channels (the spec and the delta's)
bands, frames, n_channels = 60, 41, 1
image_shape = [bands,frames,n_channels]

# First convolutional ReLU layer
n_filter_1 = 80
kernel_size_1 = [57,6]
kernel_strides_1=(1,1)
#Activation in the layer definition
#activation_1="relu"
#L2 weight decay
l2_1=0.001

#Dropout rate before pooling
dropout_1=0.5

# First MaxPool layer
pool_size_1 = (4,3)
pool_strides_1=(1,3)
padding_1="valid"

### Second convolutional ReLU layer
n_filter_2 = 80
kernel_size_2=[1,3]
kernel_strides_2=(1,1)
#Activation in the layer definition
#activation_2="relu"
l2_2=0.001

# Scond MaxPool layer
pool_size_2=(1,3)
pool_strides_2=(1,3)
padding_2="valid"

#Third (dense) ReLU layer
num_units_3 = 5000
#Activation in the layer definition
#activation_3 = "relu"
dropout_3=0.5
l2_3=0.001

#Fourth (dense) ReLU layer
num_units_4 = 5000
#Activation in the layer definition
#activation_4 = "relu"
dropout_4 = 0.5
l2_4=0.001

#Output softmax layer (10 classes in UrbanSound8K)
num_classes=10
#Activation in the layer definition
#activation_output="softmax"
l2_output=0.001

#Learning rate
learning_rate=0.01
momentum=0.9

###PLACEHOLDER VARIABLES
x_pl_1 = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')
y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
y_pl = tf.cast(y_pl, tf.float32)

##############################
###   NETWORK DEFINITION   ###
##############################

print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')

### DEEP FOURIER NETWORK
with tf.variable_scope('DF_convLayer1'):
    ### INPUT DATA
    print('x_pl_1 \t\t', x_pl_1.get_shape());
    
    ### CONV1 LAYER
    # Layer build
    z1 = tf.layers.conv1d(   inputs=x_pl_1,
                             filters=DF_filters_1,
                             kernel_size=DF_kernel_size_1,
                             strides=DF_strides_conv1,
                             padding=DF_padding_conv1,
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
                             name="DF_conv_1",
                             #reuse=None
                            )
    # Input pass, activation 
    print('DF_conv1 \t\t', z1.get_shape())                     
                            
    ### MAX_POOL1
    pool1 = tf.layers.MaxPooling1D(    pool_size=DF_pool_size_1,
                                    strides=DF_strides_pool1,
                                    padding=DF_padding_pool1, 
                                    name='DF_pool_1'
                                )
                                   
    # Activation pass, pooling
    a1 = (pool1(z1));
    print('DF_pool1 \t\t', a1.get_shape())
    
with tf.variable_scope('DF_convLayer2'):    
    ### CONV2 LAYER
    # Layer build
    z2 = tf.layers.conv1d(   inputs=a1,
                             filters=DF_filters_2,
                             kernel_size=DF_kernel_size_2,
                             strides=DF_strides_conv2,
                             padding=DF_padding_conv2,
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
                             name="DF_conv_2",
                             #reuse=None
                            )
    # Input pass, activation 
    print('DF_conv2 \t\t', z2.get_shape())                     
                            
    ### MAX_POOL2
    pool2 = tf.layers.MaxPooling1D(    pool_size=DF_pool_size_2,
                                    strides=DF_strides_pool2,
                                    padding=DF_padding_pool2, 
                                    name='DF_pool_2'
                                )
                                   
    # Activation pass, pooling
    a2 = (pool2(z2));
    # Reshaping to swtich dimension and get them right (to 41x60 to 60x41x1)
    a2 = tf.transpose(a2, perm=[0,2,1]);
    a2 = tf.expand_dims(a2, axis=3)
    #a2 = tf.reshape(a2, )
    print('DF_pool2 \t\t', a2.get_shape())
    
### PICZAK NETWORK
# Convolutional layers
with tf.variable_scope('PZ_convLayer1'):
    conv1 = tf.layers.conv2d(
        inputs=a2, ### NB!!! This is the output of the Deep_Fourier Network
        filters=n_filter_1,
        kernel_size=kernel_size_1,
        strides=kernel_strides_1,
        padding=padding_1,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_1))
    x=conv1;
    print('PZ_conv1 \t\t', x.get_shape())
    pool1 = max_pool2d(x, kernel_size=pool_size_1,stride=pool_strides_1, padding=padding_1)
    x = pool1
    print('PZ_pool1 \t\t', x.get_shape())
    x = tf.nn.dropout(x,dropout_1)

with tf.variable_scope('PZ_convLayer2'):
    conv2 = tf.layers.conv2d(
        inputs=x,
        filters=n_filter_2,
        kernel_size=kernel_size_2,
        strides=kernel_strides_2,
        padding=padding_2,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_2))
    x = conv2
    print('PZ_conv2 \t\t', x.get_shape())
    pool2 = max_pool2d(x,kernel_size=pool_size_2, stride=pool_strides_2, padding=padding_2)
    x = pool2
    print('PZ_pool2 \t\t', x.get_shape())
    # We flatten x for dense layers
    x = flatten(x)
    print('PZ_Flatten \t', x.get_shape())


# Dense layers
with tf.variable_scope('PZ_denseLayer3'):
    dense3 = tf.layers.dense(
        inputs=x,
        units=num_units_3,
        activation=tf.nn.relu,
        #kernel_initializer=None,
        #bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_3),
    )
    x = dense3
    print('PZ_dense3 \t\t', x.get_shape())
    x = tf.nn.dropout(x,dropout_3)

with tf.variable_scope('PZ_denseLayer4'):
    dense4 = tf.layers.dense(
        inputs=x,
        units=num_units_4,
        activation=tf.nn.relu,
        #kernel_initializer=None,
        #bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_4),
    )
    x = dense4
    print('PZ_dense4 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_4)

with tf.variable_scope('PZ_output_layer'):
    dense_out = tf.layers.dense(
        inputs=x,
        units=num_classes,
        activation=None,
        #kernel_initializer=None,
        #bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_output),
    )
    y = tf.nn.softmax(dense_out)
    print('denseOut\t', y.get_shape())
    
print('Model consits of ', utils.num_params(), 'trainable parameters.')

#########################################
###   SETTING VARIABLES TRAINABILITY  ###
#########################################
    
### STORING TRAINABLE VARIABLES
all_vars = list()
for variabs in tf.trainable_variables():
    all_vars.append(variabs) # Store all trainable variables in a list
    
### SLICING VARIABLES 
# Deep Fourier training variables
DF_trainables = all_vars[0:4];

# Piczak (CNN) training variables
PZ_CNN_trainables = all_vars[4:8];

# Piczak (Fully Connected) training variables
PZ_FullyC_trainables = all_vars[8:];

### SETTING TRAINABILITY FLAGS
DF_trainable = True;
PZ_CNN_firstlayer_trainable = True; #in case we want to make only the first convlayer of Piczak trainable
PZ_CNN_trainable = False;
PZ_FullyC_trainable = False;

### CREATING LIST OF VARIABLES TO TRAIN
# NB!!! Please do not put PZ_CNN_firstlayer_trainable and PZ_CNN_trainable both to True (it will break)
to_train = list();
if(DF_trainable):
    to_train.append(DF_trainables);
if(PZ_CNN_trainable):
    to_train.append(PZ_CNN_trainables)

####################################################
###   LOSS, TRAINING AND PERFORMANCE DEFINITION  ###
####################################################

with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), reduction_indices=[1])
    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.variable_scope('training'):
    # defining our optimizer
    sgd = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    # applying the gradients
    train_op = sgd.minimize(cross_entropy,  var_list=what_we_want_to_train)

with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
##############################
###   FORWARD PASS TESTING ###
##############################

#Random audio images for testing
#x_test_forward = np.random.normal(0, 1, [50,20992,1]).astype('float32') #dummy data
#
## restricting memory usage, TensorFlow is greedy and will use all memory otherwise
#gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#
#sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
#sess.run(tf.global_variables_initializer())
#feed_dict = {x_pl_1: x_test_forward}
#res_forward_pass = sess.run(fetches=[y], feed_dict=feed_dict)
#
#print("y", res_forward_pass[0].shape)
#print('Forward pass successful!')

##########################
###   TRAINING LOOP    ###
##########################
