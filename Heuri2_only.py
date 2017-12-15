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

n_fold = 10
# Folds we validate and test on (only relevant if RUN_CV==False). Indices of the folds are in the "Matlab" naming mode
k_valid = 9
k_test = 10
# If we just want to test quickly on a few epochs
RUN_FAST = True
# If we want oversampled (balanced) mini-batches or not
BALANCED_BATCHES = False
# Learning rate
learning_rate = 0.0003
# Number of epochs (only one is relevant acc. to which RUN_FAST value has been given)
max_epochs_fast = 2
max_epochs_regular = 200
# Batch size
batch_size = 100

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
print('Learning rate : {0}'.format(learning_rate))

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

if BALANCED_BATCHES:
    word_bal = 'bal'
else:
    word_bal = 'unbal'

word_lr = str(learning_rate)
word_lr = word_lr[:1] + '-' + word_lr[2:]

# Filename typesetting:
# deepFourier_df{trainable}_dfcnn1{init}_dfcnn2{init}_pzcnn1{trainable}{init}_pzcnn2{trainable}{init}_pzfc1{trainable}{init}_pzfc2{trainable}{init}_pzout{init}_{word_cv}_{word_bal}_LR{word_lr}_ME{max_epochs}
filename = "Heuri2_only_{0}_{1}_LR{2}_ME{3}".format(word_cv, word_bal, word_lr, max_epochs)

# %%
##############################
###   NETWORK PARAMETERS   ###
##############################

###GENERAL VARIABLES
# Input data size
length = 20992  # k0
nchannels = 1  # ??? Maybe two because stereo audiofiles?

###DEEP_FOURIER LAYERS HYPERPARAMETERS
# Conv1, MaxPool1 parameters
DF_padding_conv1 = "same"
DF_filters_1 = 80  # phi1
DF_kernel_size_1 = 1024
DF_strides_conv1 = 512
# DF_padding_pool1 = "valid"
# DF_pool_size_1 = 9
# DF_strides_pool1 = 3

# Conv2, MaxPool2 parameters
DF_padding_conv2 = 'same'
DF_filters_2 = 60  # phi2
DF_kernel_size_2 = 15
DF_strides_conv2 = 1
# DF_padding_pool2 = 'valid'
# DF_pool_size_2 = 9
# DF_strides_pool2 = 4

###PICZACK HYPERPARAMETERS
# Bands : related to frequencies. Frames : related to audio sequence. 2 channels (the spec and the delta's)
bands, frames, n_channels = 60, 41, 1
image_shape = [bands, frames]

# Learning rate
momentum = 0.9

###PLACEHOLDER VARIABLES
x_pl = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')
y_pl = tf.placeholder(tf.float64, [None, bands, frames], name='yPlaceholder')
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
    z1 = tf.layers.conv1d(inputs=x_pl,
                          filters=DF_filters_1,
                          kernel_size=DF_kernel_size_1,
                          strides=DF_strides_conv1,
                          padding=DF_padding_conv1,
                          # data_format='channels_last',
                          # dilation_rate=1,
                          activation=tf.abs,
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

    # ### MAX_POOL1
    # pool1 = MaxPooling1D(pool_size=DF_pool_size_1,
    #                      strides=DF_strides_pool1,
    #                      padding=DF_padding_pool1,
    #                      name='DF_pool_1'
    #                      )

    # # Activation pass, pooling
    # a1 = (pool1(z1))
    # print('DF_pool1 \t\t', a1.get_shape())

with tf.variable_scope('DF_convLayer2'):
    ### CONV2 LAYER
    print("--- Deep Fourier conv layer 2")
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

    # ### MAX_POOL2
    # pool2 = MaxPooling1D(pool_size=DF_pool_size_2,
    #                      strides=DF_strides_pool2,
    #                      padding=DF_padding_pool2,
    #                      name='DF_pool_2'
    #                      )

    # # Activation pass, pooling
    # a2 = (pool2(z2))
    # # Reshaping to swtich dimension and get them right (to 41x60 to 60x41x1)
    # a2 = tf.transpose(a2, perm=[0, 2, 1])
    # a2 = tf.expand_dims(a2, axis=3)
    # # a2 = tf.reshape(a2, )
    # print('DF_pool2 \t\t', a2.get_shape())
    y=tf.transpose(z2,perm=[0,2,1])
    print('Output \t\t',y.get_shape())
print('Model consists of ', utils.num_params(), 'trainable parameters.')
# %%
#########################################
###   SETTING VARIABLES TRAINABILITY  ###
#########################################

### STORING TRAINABLE VARIABLES
all_vars = tf.trainable_variables()

### SLICING VARIABLES
# Deep Fourier training variables
DF_trainable_stuff = all_vars[0:4]

### CREATING LIST OF VARIABLES TO TRAIN
to_train = list()
to_train += DF_trainable_stuff

print("and we will train: ")
for j in range(len(to_train)):
    print("## ", to_train[j])

####################################################
###   LOSS, TRAINING AND PERFORMANCE DEFINITION  ###
####################################################

with tf.variable_scope('loss'):
    #Computing the sum of squares error
    mean_squared_error=tf.losses.mean_squared_error(
        y_pl,
        y,
        weights=1.0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES
    )

with tf.variable_scope('training'):
    # defining our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # applying the gradients
    train_op = optimizer.minimize(mean_squared_error, var_list=to_train)

# %%
##############################
###   FORWARD PASS TESTING ###
##############################

# Random audio images for testing
x_test_forward = np.random.normal(0, 1, [50, 20992, 1]).astype('float32')  # dummy data
y_dummy_train = np.random.normal(0, 1, [50, bands, frames]).astype('float32')

# This hell line
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

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
fold_spcgm_max_vals = np.zeros(10, np.float64)
fold_spcgm_min_vals = np.zeros(10, np.float64)
fold_spcgm_mean_vals = np.zeros(10, np.float64)
for i in range(1, 11):
    data_mat = scipy.io.loadmat(data_folder + 'fold{}_wav.mat'.format(i))
    # Add one dimension for being eligible for the network
    data_mat = np.expand_dims(data_mat['ob_wav'], axis=-1)
    data_folds.append(data_mat)
    labels_mat = scipy.io.loadmat(data_folder + 'fold{}_spcgm.mat'.format(i))
    fold_spcgm_max_vals[i-1] = np.max(labels_mat['ob_spcgm']) #max over the entire fold, entire data (all 3 dimensions)
    fold_spcgm_min_vals[i-1] = np.min(labels_mat['ob_spcgm'])
    fold_spcgm_mean_vals[i - 1] = np.mean(labels_mat['ob_spcgm'])
    #labels_mat = np.expand_dims(labels_mat['ob_spcgm'], axis=-1)
    labels_folds.append(labels_mat['ob_spcgm'])

mean_over_all_folds = np.mean(fold_spcgm_mean_vals)
max_over_all_folds = np.max(fold_spcgm_max_vals)
min_over_all_folds = np.min(fold_spcgm_min_vals)

#Centering around the min and division by the range
for i in range(10):
    labels_folds[i]=(labels_folds[i]-mean_over_all_folds)/(max_over_all_folds-min_over_all_folds)
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
        merged_train_labels = np.empty((0, bands, frames))
        for i_merge in range(n_fold - 2):
            merged_train_data = np.vstack((merged_train_data, train_data[i_merge]))
            merged_train_labels = np.vstack((merged_train_labels, train_labels[i_merge]))

        # ### COMMENT THIS OUT AND UNCOMMENT ABOVE FOR FULL TRAINING DATA
        #        i_merge = 1;
        #        merged_train_data = np.vstack((merged_train_data, train_data[i_merge]))
        #        merged_train_labels = np.vstack((merged_train_labels, train_labels[i_merge]))

        train_data = merged_train_data
        train_labels = merged_train_labels

        train_loader = bl.batch_loader(train_data, train_labels, batch_size, is_balanced=BALANCED_BATCHES,
                                       is_fast=RUN_FAST)

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
            fetches_train = [train_op, mean_squared_error]
            # running the train_op and computing the updated training loss and accuracy
            res = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)
            # storing cross entropy (second fetch argument, so index=1)
            _train_loss += [res[1]]

            ### VALIDATING ###
            # When we reach the last mini-batch of the epoch
            if train_loader.is_epoch_done():
                # what to feed our accuracy op
                feed_dict_valid = {x_pl: valid_data, y_pl: valid_labels}
                # deciding which parts to fetch
                fetches_valid = [mean_squared_error]
                # running the validation
                res = sess.run(fetches=fetches_valid, feed_dict=feed_dict_valid)
                # Update all accuracies
                valid_loss += [res[0]]
                train_loss += [np.mean(_train_loss)]
                # Balanced validation accuracy
                _train_loss = []
                # Print a summary of the training and validation
                print(
                    "Epoch {} : Train Loss {:6.3f}, Valid loss {:6.3f}, took {:10.2f} sec".format(
                        epoch, train_loss[-1], valid_loss[-1],
                        time.time() - TIME_epoch_start))
                print("")
                TIME_epoch_start = time.time()
                # "Early stopping" (in fact, we keep going but just take the best network at every time step we have improvement)
                if valid_loss[-1] == max(valid_loss):
                    # Updating the best quantities
                    best_train_loss = train_loss[-1]
                    best_valid_loss = valid_loss[-1]
                    best_epoch = epoch

                    # Weights
                    variables_names = [v.name for v in tf.trainable_variables()]
                    best_weights = sess.run(variables_names)

                # Update epoch
                epoch += 1;
        # Save everything (all training history + the best values)
        mdict = {'train_loss': train_loss, 'valid_loss': valid_loss,
                 'best_train_loss': best_train_loss,
                 'best_valid_loss': best_valid_loss,
                 'best_epoch': best_epoch,
                 }
        scipy.io.savemat(save_path_perf + filename + "_ACCURACY", mdict)

        # Saving the weights
        TIME_saving_start = time.time()
        print("performance saved under %s...." % save_path_perf)
        scipy.io.savemat(save_path_numpy_weights + filename + "_WEIGHTS", dict(
            DF_conv1d_1_kernel=best_weights[0],
            DF_conv1d_1_bias=best_weights[1],
            DF_conv1d_2_kernel=best_weights[2],
            DF_conv1d_2_bias=best_weights[3]
        ))
        print("weights (numpy arrays) saved under %s....: " % save_path_numpy_weights)
        print("... saving took {:10.2f} sec".format(time.time() - TIME_saving_start))

    except KeyboardInterrupt:
        pass

print("<><><><><><><><> the entire program finished without errors!! <><><><><><><><>")
