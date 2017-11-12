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

print('Model consits of ', utils.num_params(), 'trainable parameters.')

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

#saver = tf.train.Saver()

### Test the forward pass : ADD THE BATCH LOADER (pylearn2? test it) + understand data structure + see how to run AWS

#Random audio images for testing
x_test_forward = np.random.normal(0, 1, [1,60,41,1]).astype('float32') #dummy data

# restricting memory usage, TensorFlow is greedy and will use all memory otherwise
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
sess.run(tf.global_variables_initializer())
feed_dict = {x_pl: x_test_forward}
res_forward_pass = sess.run(fetches=[y], feed_dict=feed_dict)

print("y", res_forward_pass[0].shape)
print('Forward pass successful!')


### Training

# Batch shit
batch_size = 1000
max_epochs = 1
epoch=0

# For now, training fold
fold1mat=scipy.io.loadmat('fold1_with_irregwin.mat')
training_data=fold1mat['ob']
training_data=np.expand_dims(training_data,axis=-1)
training_labels=utils.onehot(np.transpose(fold1mat['lb']),num_classes)
training_loader = bl.batch_loader(training_data, training_labels, batch_size);

# For now, test fold
fold2mat=scipy.io.loadmat('fold2_with_irregwin.mat')
test_data=fold2mat['ob']
test_data=np.expand_dims(test_data,axis=-1)
test_labels=utils.onehot(np.transpose(fold2mat['lb']),num_classes)
test_loader = bl.batch_loader(test_data, test_labels, batch_size);

valid_loss, valid_accuracy = [], []
train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []

saver = tf.train.Saver() # defining saver function

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Begin training loop')
    try:
        while (epoch <= max_epochs):
            batch_data, batch_labels = training_loader.next_batch()
            print('====================================================')
            ### TRAINING ###
            # what to feed to our train_op
            # notice we onehot encode our predictions to change shape from (batch,) -> (batch, num_output)
            feed_dict_train = {x_pl: batch_data, y_pl: batch_labels}

            # deciding which parts to fetch, train_op makes the classifier "train"
            fetches_train = [train_op, cross_entropy, accuracy]

            # running the train_op
            res = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
            # storing cross entropy (second fetch argument, so index=1)
            train_loss += [res[1]]
            train_accuracy += [res[2]]

            ### VALIDATING ###
            batch_data, batch_labels = test_loader.next_batch();
            # what to feed our accuracy op
            feed_dict_valid = {x_pl: batch_data, y_pl : batch_labels}

            # deciding which parts to fetch
            fetches_valid = [cross_entropy, accuracy]

            # running the validation
            res = sess.run(fetches=fetches_valid, feed_dict=feed_dict_valid)
            test_loss += [res[0]]
            test_accuracy += [res[1]]

            epoch = epoch + 1;

        save_path = saver.save(sess, "/model.ckpt") # hopefully works for GBAR
        print(save_path)
    except KeyboardInterrupt:
        pass

epoch = np.arange(len(train_loss))
plt.figure()
plt.plot(epoch, train_accuracy,'r', epoch, valid_accuracy,'b')
plt.legend(['Train Acc','Val Acc'], loc=4)
plt.xlabel('Epochs'), plt.ylabel('Acc'), plt.ylim([0.75,1.03])
