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

# =============================================================================
# =============================================================================
# =============================================================================
#########################
###   CODE SETTINGS   ###
#########################
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
max_epochs_regular = 100
# Batch size
batch_size = 1000

IB_PHASE = PHASE1

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
print('Learning rate : {0}'.format(learning_rate))

# Naming of output files
# We just shift -1 the folds indices to match with Python way to think
k_valid = k_valid - 1
k_test = k_test - 1

data_folder = "./data/"

ib = icebreaker(phase=IB_PHASE, lr=learning_rate, max_epochs = max_epochs, TEST = False)
DF_arch = "Heuri1" # the DF architecture that we will use!!
if DF_arch == "Heuri1":
    DF_net_weight_names = ['DF_conv1d_1_kernel',
                                'DF_conv1d_1_bias',
                                'DF_conv1d_2_kernel',
                                'DF_conv1d_2_bias']
elif DF_arch == "HeuriSmall":
    DF_net_weight_names = ['DF_conv1d_1_kernel',
                                'DF_conv1d_1_bias']
ib.manage_weigths(DF_arch, DF_net_weight_names)
# =============================================================================
# =============================================================================
# =============================================================================





print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')
# Input data size
length = 20992  # k0
nchannels = 1  # ??? Maybe two because stereo audiofiles?
num_classes = 10
###SET RANDOM SEED AND RESET GRAPH
tf.reset_default_graph()
###PLACEHOLDER VARIABLES
x_pl = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')
y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
y_pl = tf.cast(y_pl, tf.float32)
if DF_arch == "Heuri1":
    ###DEEP_FOURIER LAYERS HYPERPARAMETERS
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

    ### DEEP FOURIER NETWORK
    with tf.variable_scope('DF_convLayer1'):
        ### INPUT DATA
        print("--- Deep Fourier conv layer 1")
        print('x_pl \t\t', x_pl.get_shape())
        ### CONV1 LAYER
        # Layer build
        ### CHECK IF WEIGHT LOADING\INITIALIZE
        if ib.shall_DF_be_loaded():
            df1_ker_init = tf.constant_initializer(ib.pretrained_DF[0])
            df1_bias_init = tf.constant_initializer(ib.pretrained_DF[1])
            print('Pretrained DF_conv1d_1 loaded!')
        else:
            df1_ker_init = df1_bias_init = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
            print('DF_conv1d_1 reinitialized with Xavier!')
        z1 = tf.layers.conv1d(inputs=x_pl,
                              filters=DF_filters_1,
                              kernel_size=DF_kernel_size_1,
                              strides=DF_strides_conv1,
                              padding=DF_padding_conv1,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=df1_ker_init,
                              bias_initializer=df1_bias_init,
                              trainable=True,
                              name="DF_conv_1",
                              )
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
            df2_ker_init = tf.constant_initializer(ib.pretrained_DF[2])
            df2_bias_init = tf.constant_initializer(ib.pretrained_DF[3])
        else:
            df2_ker_init = df2_bias_init = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        # Layer build
        z2 = tf.layers.conv1d(inputs=a1,
                              filters=DF_filters_2,
                              kernel_size=DF_kernel_size_2,
                              strides=DF_strides_conv2,
                              padding=DF_padding_conv2,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=df2_ker_init,
                              bias_initializer=df2_bias_init,
                              trainable=True,
                              name="DF_conv_2",
                              # reuse=None
                              )
        print('Pretrained DF_conv1d_2 loaded!')
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
        # Reshaping to swtich dimension and get them right (to 41x60 to 60x41x1)
        a2 = tf.transpose(a2, perm=[0, 2, 1])
        DF_OUTPUT = tf.expand_dims(a2, axis=3)
        # a2 = tf.reshape(a2, )
        print('DF_pool2 \t\t', DF_OUTPUT.get_shape())
        #very imortant!


from dimensions_PZ import *

### PICZAK NETWORK
# Convolutional layers
with tf.variable_scope('PZ_convLayer1'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    print("--- Piczak")
    conv1 = tf.layers.conv2d(
        kernel_initializer=tf.constant_initializer(ib.pretrained_PZ[0]),
        bias_initializer=tf.constant_initializer(ib.pretrained_PZ[1]),
        inputs=DF_OUTPUT,
        filters=n_filter_1,
        kernel_size=kernel_size_1,
        strides=kernel_strides_1,
        padding=padding_1,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_1))
    print('Pretrained PZ_conv2d_1 loaded!')
    x = conv1
    print('PZ_conv1 \t\t', x.get_shape())
    pool1 = max_pool2d(x, kernel_size=pool_size_1, stride=pool_strides_1, padding=padding_1)
    x = pool1
    print('PZ_pool1 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_1)

with tf.variable_scope('PZ_convLayer2'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    conv2 = tf.layers.conv2d(
        kernel_initializer=tf.constant_initializer(ib.pretrained_PZ[2]),
        bias_initializer=tf.constant_initializer(ib.pretrained_PZ[3]),
        inputs=x,
        filters=n_filter_2,
        kernel_size=kernel_size_2,
        strides=kernel_strides_2,
        padding=padding_2,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_2))
    print('Pretrained PZ_conv2d_2 loaded!')
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
    dense3 = tf.layers.dense(
        kernel_initializer=tf.constant_initializer(ib.pretrained_PZ[4]),
        bias_initializer=tf.constant_initializer(ib.pretrained_PZ[5]),
        inputs=x,
        units=num_units_3,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_3) )
    print('Pretrained PZ_dense_1 loaded!')
    x = dense3
    print('PZ_dense3 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_3)

with tf.variable_scope('PZ_denseLayer4'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    dense4 = tf.layers.dense(
        kernel_initializer=tf.constant_initializer(ib.pretrained_PZ[6]),
        bias_initializer=tf.constant_initializer(ib.pretrained_PZ[7]),
        inputs=x,
        units=num_units_4,
        activation=tf.nn.relu,
        # kernel_initializer=None,
        # bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_4),
    )
    print('Pretrained PZ_dense_2 loaded!')
    print('PZ_dense4 \t\t', x.get_shape())
    x = tf.nn.dropout(x, dropout_4)

with tf.variable_scope('PZ_output_layer'):
    ### CHECK IF WEIGHT LOADING\INITIALIZE
    dense_out = tf.layers.dense(
        kernel_initializer=tf.constant_initializer(ib.pretrained_PZ[8]),
        bias_initializer=tf.constant_initializer(ib.pretrained_PZ[9]),
        inputs=x,
        units=num_classes,
        activation=None,
        # kernel_initializer=None,
        # bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_output),
    )
    print('Pretrained PZ_output loaded!')
    y = tf.nn.softmax(dense_out)
    print('denseOut\t', y.get_shape())

print('Model consists of ', utils.num_params(), 'trainable parameters.')

# %%
#########################################
###   SETTING VARIABLES TRAINABILITY  ###
#########################################

### STORING TRAINABLE VARIABLES
all_vars = tf.trainable_variables()
to_train = ib.what_is_trainable(all_vars)
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

with tf.variable_scope('training'):
    # defining our optimizer
    sgd = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    # applying the gradients
    train_op = sgd.minimize(cross_entropy, var_list=to_train)

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
    if RUN_FAST: labels_mat = utils.onehot(np.random.randint(0, 10, small_data), 10)
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
        epoch = 0

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
        valid_loss, valid_accuracy = [], []
        bal_valid_accuracy = []
        test_loss, test_accuracy = [], []

        ### TRAINING ###
        TIME_epoch_start = time.time()
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
                        epoch, train_loss[-1], train_accuracy[-1], valid_loss[-1], bal_valid_accuracy[-1],
                        time.time() - TIME_epoch_start))
                print("")
                TIME_epoch_start = time.time()
                # "Early stopping" (in fact, we keep going but just take the best network at every time step we have improvement)
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
                 'valid_accuracy': valid_accuracy, 
                 'bal_valid_accuracy': bal_valid_accuracy, 'best_bal_train_loss': best_bal_train_loss,
                 'best_bal_train_accuracy': best_bal_train_accuracy, 'best_bal_valid_loss': best_bal_valid_loss,
                 'best_bal_valid_accuracy': best_bal_valid_accuracy, 'best_bal_epoch': best_bal_epoch}
        scipy.io.savemat(ib.good_place_to_store_perf() + "_ACCURACY", mdict)
        print("'balanced' weights (numpy arrays) saved under %s....: " % (ib.good_place_to_store_perf() + "_ACCURACY") )
        ib.save_weights(best_bal_weights)

    except KeyboardInterrupt:
        pass

print("<><><><><><><><> the entire program finished without errors!! <><><><><><><><>")
