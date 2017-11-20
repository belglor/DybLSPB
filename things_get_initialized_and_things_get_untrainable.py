from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import os
import sys
sys.path.append(os.path.join('.', '..'))
import utils
import batch_loader as bl
import scipy
from scipy import io
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn

# a dummy kernel to inialize one layer of the network, just to see if it works. Later this will be the loaded kernels from the trained piczak network
loaded_outputlayer_kernel = np.array( [ np.arange(0, 50, .01) ] )
loaded_outputlayer_kernel = loaded_outputlayer_kernel.repeat(10, axis=0)
loaded_outputlayer_kernel = loaded_outputlayer_kernel.T
loaded_outputlayer_kernel2 = -1337*np.ones((5000, 10), np.float32)
#loaded_outputlayer_kernel_as_tensor = tf.Variable(loaded_outputlayer_kernel, name="loaded", dtype=tf.float32) 
loaded_outputlayer_kernel_as_tensor = tf.constant_initializer(loaded_outputlayer_kernel2)
WHERE_IS_OUTPUTLAYER_KERNEL = 8

tf.reset_default_graph()

#bands : related to frequencies. Frames : related to audio sequence. 2 channels (the spec and the delta's)
bands, frames, n_channels = 60, 41, 1
image_shape = [bands,frames,n_channels]

### First convolutional ReLU layer
n_filter_1 = 80
kernel_size_1 = [57,6]
kernel_strides_1=(1,1)
#Activation in the layer definition
#activation_1="relu"
#L2 weight decay
l2_1=0.001

#Dropout rate before pooling
dropout_1=0.5

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
learning_rate=0.5 #just to make things more visible 0.001
momentum=0.9

tf.reset_default_graph()

#Setting up the placeholders
x_pl = tf.placeholder(tf.float32, [None, bands, frames,n_channels], name='xPlaceholder')
y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
y_pl = tf.cast(y_pl, tf.float32)

#Setting up the filters and weights to optimize
print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')

# Convolutional layers
with tf.variable_scope('convLayer1'):
    conv1 = tf.layers.conv2d(
        inputs=x_pl,
        filters=n_filter_1,
        kernel_size=kernel_size_1,
        strides=kernel_strides_1,
        padding=padding_1,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_1))
    x=conv1
    print('x_pl \t\t', x_pl.get_shape())
    #Just define successive values with x in case we add intermediate layers
    print('conv1 \t\t', x.get_shape())
    pool1 = max_pool2d(x, kernel_size=pool_size_1,stride=pool_strides_1, padding=padding_1)
    #How to add dropout for pooling?
    x = pool1
    print('pool1 \t\t', x.get_shape())
    x = tf.nn.dropout(x,dropout_1)

with tf.variable_scope('convLayer2'):
    conv2 = tf.layers.conv2d(
        inputs=x,
        filters=n_filter_2,
        kernel_size=kernel_size_2,
        strides=kernel_strides_2,
        padding=padding_2,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_2))
    x = conv2
    print('conv2 \t\t', x.get_shape())
    pool2 = max_pool2d(x,kernel_size=pool_size_2, stride=pool_strides_2, padding=padding_2)
    x = pool2
    print('pool2 \t\t', x.get_shape())
    # We flatten x for dense layers
    x = flatten(x)
    print('Flatten \t', x.get_shape())


# Dense layers
with tf.variable_scope('denseLayer3'):
    dense3 = tf.layers.dense(
        inputs=x,
        units=num_units_3,
        activation=tf.nn.relu,
        #kernel_initializer=None,
        #bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_3),
    )
    x = dense3
    print('dense3 \t\t', x.get_shape())
    x = tf.nn.dropout(x,dropout_3)

with tf.variable_scope('denseLayer4'):
    dense4 = tf.layers.dense(
        inputs=x,
        units=num_units_4,
        activation=tf.nn.relu,
        #kernel_initializer=None,
        #bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_4),
    )
    x = dense4
    print('dense4 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_4)

with tf.variable_scope('output_layer'):
    dense_out = tf.layers.dense(
        inputs=x,
        units=num_classes,
        activation=None,
        #kernel_initializer=None,
        #bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_output),
        kernel_initializer = loaded_outputlayer_kernel_as_tensor
    )
    y = tf.nn.softmax(dense_out)
    print('denseOut\t', y.get_shape())

print('Model consists of ', utils.num_params(), 'trainable parameters.')

with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), reduction_indices=[1])
    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)

what_we_want_to_train = list()
for stuff in tf.trainable_variables()[0:2]: #we want to update only the first Piczak Layer (Kernel and Bias, ergo 2 elements in the list) 
    what_we_want_to_train.append(stuff)
print("what_we_want_to_train: ", what_we_want_to_train)                                              

with tf.variable_scope('training'):
    # defining our optimizer
    sgd = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    # applying the gradients
    train_op_full = sgd.minimize(cross_entropy, name="IWillUpdateAllTheNetwork")
    train_op_partial = sgd.minimize(cross_entropy, var_list=what_we_want_to_train, name="IWillUpdateOnlyTheWantedLayers")

with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

np.random.seed(43) # 42 is mainstream
x_dummy = np.random.normal(0, 1, [1337,60,41,1]).astype('float32') #dummy data
y_dummy = utils.onehot(np.random.uniform(0, 10, [1337]).astype('int32'), 10 ) #dummy labels
# please use the SAME random data and labels in all the 4 sessions below, for comparison 

TF_RAND_SEED = 666 # needed, since TF cares not about the numpy random seed above
tf.set_random_seed(TF_RAND_SEED) #we have still no idea how to use this correctly, ask the TA!
print("")
print("+++++++++++++++++++++++++++++++++++")
with tf.Session() as the_session:
    print("######### DUMMY TRAINING")
    print("-------------------------------")
    print("---when we update all weights:")
    print("-------------------------------")
    tf.set_random_seed(TF_RAND_SEED)
    the_session.run(tf.global_variables_initializer())
    fetches_train = [train_op_full, cross_entropy]
    print("the output layer kernel before training op run", the_session.run(tf.trainable_variables()[6].name) )
    for _ in range(3): the_session.run(fetches=fetches_train, feed_dict={x_pl: x_dummy, y_pl:y_dummy})
    print("")
    print("the output layer kernel AFTER training op run (should have changed)", the_session.run(tf.trainable_variables()[6].name) )
    print("")
    print("-------------------------------")
    print("---when we update only the wanted weights:")
    print("-------------------------------")
    tf.set_random_seed(TF_RAND_SEED)
    the_session.run(tf.global_variables_initializer())
    fetches_train = [train_op_partial, cross_entropy]
    print("the output layer kernel before training op run", the_session.run(tf.trainable_variables()[6].name) )
    for _ in range(3): the_session.run(fetches=fetches_train, feed_dict={x_pl: x_dummy, y_pl:y_dummy})
    print("")
    print("the output layer kernel AFTER training op run (should NOT have changed)", the_session.run(tf.trainable_variables()[6].name) )

                                                     