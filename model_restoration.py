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

tf.reset_default_graph()

# Random seed for reproducibility
np.random.seed(1337)

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
learning_rate=0.001
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
    )
    y = tf.nn.softmax(dense_out)
    print('denseOut\t', y.get_shape())

print('Model consists of ', utils.num_params(), 'trainable parameters.')

with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), reduction_indices=[1])
    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.variable_scope('training'):
    # defining our optimizer
    sgd = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    # applying the gradients
    train_op = sgd.minimize(cross_entropy)

with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

save_path="./saved_models/piczak_150.ckpt-150"
with tf.Session() as first_restore_session:
    saver.restore(first_restore_session, save_path)
    tf.trainable_variables()
    # Check some variables from loaded model
    variables_names =[v.name for v in tf.trainable_variables()]   # get all trainable shit from piczak
    var_value= first_restore_session.run(variables_names)                           # run them through session and save value
    index = 0
    for k,v in zip(variables_names, var_value):
        print("---------trainable stuff {0}: {1}------------------".format(index, k))
        print(v) #e.g. the saved weights/kernel elements
        print("------------------------------------------------------------------")
        print("")
        print("")
        index+=1
    what_we_want_to_train = list()
    for stuff in tf.trainable_variables()[0:2]: #we want to update only the first Piczak Layer (Kernel and Bias, ergo 2 elements in the list) 
        what_we_want_to_train.append(stuff)

print("from now on, only this will be updated: ", what_we_want_to_train)

np.random.seed(43) # 42 is mainstream
x_dummy = np.random.normal(0, 1, [1337,60,41,1]).astype('float32') #dummy data
y_dummy = utils.onehot(np.random.uniform(0, 10, [1337]).astype('int32'), 10 ) #dummy labels
# please use the SAME random data and labels in all the 4 sessions below, for comparison 

TF_RAND_SEED = 666 # needed, since TF cares not about the numpy random seed above
tf.set_random_seed(TF_RAND_SEED)
print("")
print("+++++++++++++++++++++++++++++++++++")
print("fully trainable network, run (1/2)")
with tf.Session() as uber_session:
    saver.restore(uber_session, save_path)
    tf.set_random_seed(TF_RAND_SEED)
    fetches_train = [train_op, cross_entropy, accuracy]
    res = uber_session.run(fetches=fetches_train, feed_dict={x_pl: x_dummy, y_pl:y_dummy})
    print("res train_op: ", res[0])
    print("res cross_entropy: ", res[1])
    print("res accuracy: ", res[2])
    
print("")
print("do the exact same thing again, to check whether the TF random seeding works")
print("+++++++++++++++++++++++++++++++++++")
print("fully trainable network, run (2/2)")    
with tf.Session() as uber_session2:
    saver.restore(uber_session2, save_path)
    tf.set_random_seed(TF_RAND_SEED)
    fetches_train = [train_op, cross_entropy, accuracy]
    res = uber_session2.run(fetches=fetches_train, feed_dict={x_pl: x_dummy, y_pl:y_dummy})
    print("res train_op: ", res[0])
    print("if this is now different, the random seeding does NOT work the way we want")
    print("res cross_entropy: ", res[1])
    print("res accuracy: ", res[2])

print("")
print("+++++++++++++++++++++++++++++++++++")
print("network with only first layer trainable, run (1/2)")
with tf.Session() as uber_session:
    saver.restore(uber_session, save_path)
    tf.set_random_seed(TF_RAND_SEED)
    train_op.var_list = what_we_want_to_train #does this line solve our problem?? if not, how to do it properly?? ask the TA
    fetches_train = [train_op, cross_entropy, accuracy]
    res = uber_session.run(fetches=fetches_train, feed_dict={x_pl: x_dummy, y_pl:y_dummy})
    print("res train_op: ", res[0])
    print("res cross_entropy: ", res[1])
    print("res accuracy: ", res[2])
    
print("")
print("do the exact same thing again, to check whether the TF random seeding works")
print("+++++++++++++++++++++++++++++++++++")
print("network with only first layer trainable, run (2/2)")    
with tf.Session() as uber_session2:
    saver.restore(uber_session2, save_path)
    tf.set_random_seed(TF_RAND_SEED)
    train_op.var_list = what_we_want_to_train #does this line solve our problem?? if not, how to do it properly?? ask the TA
    fetches_train = [train_op, cross_entropy, accuracy]
    res = uber_session2.run(fetches=fetches_train, feed_dict={x_pl: x_dummy, y_pl:y_dummy})
    print("res train_op: ", res[0])
    print("res cross_entropy: ", res[1])
    print("res accuracy: ", res[2])

################################################
############### LOOK AT ME #####################
################################################   

#the list 
#what_we_want_to_train
#can normally (when no session restoring is done) be given to the function
#sgd.minimize(),
#like in:
#sgd.minimize(cross_entropy, var_list = what_we_want_to_train)

#but how to do it if session restoring is used??? ask the TA!

################################################
############### THANKS FOR LOOKING! ############
################################################   
 
                                                     