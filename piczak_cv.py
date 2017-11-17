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

from sklearn.model_selection import StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

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

### Test the forward pass

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


### Cross-validation

# Batch shit
batch_size = 1000
max_epochs = 1
epoch=0

#Folds creation : lists with all the np.array's inside
data_folds=[]
labels_folds=[]
for i in range(1,11):
    data_mat=scipy.io.loadmat('fold{}_spcgm.mat'.format(i))
    #Add one dimension for being eligible for the network
    data_mat=np.expand_dims(data_mat['ob_spcgm'],axis=-1)
    data_folds.append(data_mat)
    labels_mat=scipy.io.loadmat('fold{}_labels.mat'.format(i))
    labels_mat=utils.onehot(np.transpose(labels_mat['lb']), num_classes)
    #One-hot encoding labels
    labels_folds.append(labels_mat)

#Training loss and accuracy for each epoch
train_loss, train_accuracy = [], []
#Training loss and accuracy within an epoch (is erased at every new epoch)
_train_loss, _train_accuracy = [], []
valid_loss, valid_accuracy = [], []
test_loss, test_accuracy = [], []
saver = tf.train.Saver() # defining saver function

#Cross validation parameters
n_fold=10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Cross-validation
    #for cv in range(n_fold):

    print('Begin training loop')
    try:
        #Cross-validation
        for k in range(n_fold):
            mask=[True]*n_fold
            mask[k]=False
            train_data=[data_folds[i] for i in range(len(mask)) if mask[i]]
            train_labels=[labels_folds[i] for i in range(len(mask)) if mask[i]]
            train_loader = bl.batch_loader(train_data, train_labels, batch_size)

        ### TRAINING ###
        while (epoch < max_epochs):
            train_batch_data, train_batch_labels = train_loader.next_batch()
            feed_dict_train = {x_pl: train_batch_data, y_pl: train_batch_labels}
            # deciding which parts to fetch, train_op makes the classifier "train"
            fetches_train = [train_op, cross_entropy, accuracy]
            # running the train_op and computing the updated training loss and accuracy
            res = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
            # storing cross entropy (second fetch argument, so index=1)
            _train_loss += [res[1]]
            _train_accuracy += [res[2]]

            ### VALIDATING ###
            #When we reach the last mini-batch of the epoch
            if train_loader.is_epoch_done():
                # what to feed our accuracy op
                feed_dict_valid = {x_pl: valid_data, y_pl : valid_labels}
                # deciding which parts to fetch
                fetches_valid = [cross_entropy, accuracy]
                # running the validation
                res = sess.run(fetches=fetches_valid, feed_dict=feed_dict_valid)
                #Update all accuracies
                valid_loss += [res[0]]
                valid_accuracy += [res[1]]
                train_loss+=[np.mean(_train_loss)]
                train_accuracy+=[np.mean(_train_accuracy)]
                # Reinitialize the intermediate loss and accuracy within epochs
                _train_loss, _train_accuracy = [], []
                #Print a summary of the training and validation
                print("Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}".format(
                    epoch, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1]))

                #Update epoch
                epoch += 1;
        #Test the training on a independent test set (relevant when we'll tune parameters using the validation set/do early stopping)
        test_epoch = epoch
        # while mnist_data.test.epochs_completed == test_epoch:
        #     x_batch, y_batch = mnist_data.test.next_batch(batch_size)
        #     feed_dict_test = {x_pl: x_batch, y_pl: y_batch}
        #     _loss, _acc = sess.run(fetches_valid, feed_dict_test)
        #     test_loss.append(_loss)
        #     test_accuracy.append(_acc)

        whatever = 1000
        save_path = saver.save(sess, "./saved_models/the_saved_model.ckpt", global_step=whatever) # hopefully works for GBAR and on your local computer
        print("model saved under the path: ", save_path)
    except KeyboardInterrupt:
        pass

epoch = np.arange(len(train_loss))
#TODO: fix this
#plt.figure()
#plt.plot(epoch, train_accuracy,'r', epoch, valid_accuracy,'b')
#plt.legend(['Train Acc','Val Acc'], loc=4)
#plt.xlabel('Epochs'), plt.ylabel('Acc'), plt.ylim([0.75,1.03])
