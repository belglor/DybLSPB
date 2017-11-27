from __future__ import absolute_import, division, print_function
import sys

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import os
sys.path.append(os.path.join('.', '..'))
import utils
import batch_loader as bl
import scipy
from scipy import io
import time
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn
########################################
n_fold=10
if len(sys.argv) == 2:       #if you run the script in the CMD/shell like 
                             # python3 piczak_cv.py FAST
    if sys.argv[1] == "FAST": 
        RUN_FAST = True
        CV_VALID_FOLDS = [1, 6]
        print("running in fast mode (not much data).")
    else: 
        RUN_FAST = False
        try: CV_VALID_FOLDS = [int(sys.argv[1])] # python3 piczak_cv.py 3 --> test only on fold 3
        except ValueError: CV_VALID_FOLDS = range(n_fold)
else: #if you run the script "normally" from Pycharm, Spyder or shell 
    RUN_FAST = False
    # True: run only with a few data and not all CV folds, just to check
    # False [default]: run everything (same behaviour as before)  
    CV_VALID_FOLDS = range(n_fold)
#Cross validation parameter
if RUN_FAST: 
    max_epochs = 2
else: 
    max_epochs = 300
batch_size = 1000
directory = "./trained_models/piczak_{0}/".format(max_epochs)
save_path_perf = directory + "performance"
save_path_numpy_weights = directory + "trainedweights"
if RUN_FAST: save_path_numpy_weights += "_FAST"

#for the time being, hardcoded
CLEAN = True

########################################
print("CV: will validate on folds ", CV_VALID_FOLDS)

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
learning_rate=0.002
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
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
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
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
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
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
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
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
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
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
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

#Folds creation : lists with all the np.array's inside
data_folds=[]
labels_folds=[]
for i in range(1,11):
    data_mat=scipy.io.loadmat('fold{}_spcgm.mat'.format(i))
    #Add one dimension for being eligible for the network
    data_mat=np.expand_dims(data_mat['ob_spcgm'],axis=-1)
    data_folds.append(data_mat)
    labels_mat=scipy.io.loadmat('fold{}_labels.mat'.format(i))
    labels_mat=utils.onehot(np.transpose(labels_mat['lb']), num_classes) #One-hot encoding labels
    labels_folds.append(labels_mat)

saver = tf.train.Saver() # defining saver function
#Initialization of values over folds
#I commented this out because we will now save one mat file for every fold
#train_loss_cv = np.zeros((n_fold,max_epochs))
#train_accuracy_cv = np.zeros((n_fold,max_epochs))
#valid_loss_cv = np.zeros((n_fold,max_epochs))
#valid_accuracy_cv = np.zeros((n_fold,max_epochs))
with tf.Session() as sess:
    #Cross-validation
    try:
        #Cross-validation loop
        for k in CV_VALID_FOLDS:
            foldname = "_CVValidFold%d"%k
            print("------------------------------------------------------------")
            print('----training on all folds but no. {0}. Validating on no. {0}'.format(k+1))
            print("------------------------------------------------------------")
            #We reinitialize the weights
            sess.run(tf.global_variables_initializer())
            epoch = 0

            mask=[True]*n_fold
            mask[k]=False
            train_data=[data_folds[i] for i in range(len(mask)) if mask[i]]
            train_labels=[labels_folds[i] for i in range(len(mask)) if mask[i]]
            #Merging data (list being different from np.arrays)
            merged_train_data=np.empty((0,bands,frames,n_channels))
            merged_train_labels=np.empty((0,num_classes))
            for i_merge in range(n_fold-1):
                merged_train_data=np.append(merged_train_data,data_folds[i_merge],axis=0)
                merged_train_labels=np.append(merged_train_labels,labels_folds[i_merge],axis=0)

            train_data=merged_train_data
            train_labels=merged_train_labels

            train_loader = bl.batch_loader(train_data, train_labels, batch_size, is_balanced = CLEAN, is_fast = RUN_FAST)

            valid_data=data_folds[k]
            valid_labels=labels_folds[k]

            # Training loss and accuracy for each epoch : initialization
            train_loss, train_accuracy = [], []
            # Training loss and accuracy within an epoch (is erased at every new epoch)
            _train_loss, _train_accuracy = [], []
            valid_loss, valid_accuracy = [], []
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
                    print("Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}, took {:10.2f} sec".format(
                        epoch, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1], time.time() - TIME_epoch_start))
                    print("")
                    TIME_epoch_start = time.time()
                    #Update the cross validation results
                    #I commented this out because we will now save one mat file for every fold
#                    train_loss_cv[k,epoch]=train_loss[-1]
#                    train_accuracy_cv[k,epoch]=train_accuracy[-1]
#                    valid_loss_cv[k,epoch]=valid_loss[-1]
#                    valid_accuracy_cv[k,epoch]=valid_accuracy[-1]

                    #"Early stopping" (in fact, we keep going but just take the best network at every time step we have improvement)
                    if valid_accuracy[-1]==max(valid_accuracy):
                        pred_labels = np.argmax(sess.run(fetches=y, feed_dict={x_pl: valid_data}), axis=1)
                        true_labels = utils.onehot_inverse(valid_labels)
                        conf_mat = confusion_matrix(true_labels, pred_labels, labels=range(10))
                        print("confusion matrix for entire validation data:")
                        print(conf_mat)
                        TIME_saving_start = time.time()
                        print("saving ...")
                        save_path = saver.save(sess, directory + "TF_ckpt" + foldname + "/piczak_300.ckpt",global_step=300)  # For space issues, we indicate global step being 300 but in fact it is the best_epoch step
                        print("model TF ckpt saved under the path: ", save_path)
                        best_valid_accuracy = valid_accuracy[-1]
                        variables_names =[v.name for v in tf.trainable_variables()]   
                        var_value=sess.run(variables_names)
			           #TF saving done, now saving the convenient stuff
			           #mdict={'train_loss_cv':train_loss_cv,'train_accuracy_cv':train_accuracy_cv,'valid_loss_cv':valid_loss_cv,'valid_accuracy_cv':valid_accuracy_cv}
                        mdict={'train_loss':train_loss,'train_accuracy':train_accuracy,'valid_loss':valid_loss,'valid_accuracy':valid_accuracy,'best_epoch':epoch,'best_valid_accuracy':best_valid_accuracy}
                        if not RUN_FAST: scipy.io.savemat(save_path_perf + foldname, mdict)

                        print("performance saved under the path: ", save_path_perf)
                        scipy.io.savemat(save_path_numpy_weights + foldname, dict(
                                        conv2d_1_kernel = var_value[0],
                                        conv2d_1_bias = var_value[1],
                                        conv2d_2_kernel = var_value[2],
                                        conv2d_2_bias = var_value[3],
                                        dense_1_kernel = var_value[4],
                                        dense_1_bias = var_value[5],
                                        dense_2_kernel = var_value[6],
                                        dense_2_bias = var_value[7],
                                        output_kernel = var_value[8],
                                        output_bias = var_value[9]
                                        ) )
                        print("weights (numpy arrays) saved under the path: ", save_path_numpy_weights)
                        print("... saving took {:10.2f}".format(time.time() - TIME_saving_start))
                    # Update epoch
                    epoch += 1;
    except KeyboardInterrupt:
        pass
print("<><><><><><><><> the entire program finished without errors!! <><><><><><><><>")
