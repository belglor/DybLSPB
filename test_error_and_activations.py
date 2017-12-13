#####################
###   CODE SETUP  ###
#####################
    
from __future__ import absolute_import, division, print_function 
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import sys
sys.path.append(os.path.join('.', '..')) 
import utils 
import scipy.io
import batch_loader as bl
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.layers.pooling import MaxPooling1D
from sklearn.metrics import confusion_matrix

###SET RANDOM SEED AND RESET GRAPH

tf.reset_default_graph()

# =============================================================================
learned_weights_file = "Heuri1_OLA_L_LR0-002_ME100" #<-- the experiment name of interest
tw = scipy.io.loadmat("./results_mat/trainedweights/" + learned_weights_file + "_BAL_WEIGHTS.mat")

#if one of them is True, we include the DF part as well, because we calculate the test error on the whole DF+PZ net
include_DF_HeurNet = True
include_DF_MST = False 
# =============================================================================
data_folder = "./data/"

ONLY_EXAMPLE_OBSERVATIONS = True #if True, then it is way faster and of course the test error makes no sense
STORE_ACTIVATIONS = True
STORE_TEST_ERROR = False
#provide a filename extension here!
if ONLY_EXAMPLE_OBSERVATIONS: learned_weights_file += "_RANDOM_OBS"
if STORE_TEST_ERROR: save_path_perf = "./results_mat/performance/" + learned_weights_file + "_TESTERROR.mat"
if STORE_ACTIVATIONS: save_path_activ = "./results_mat/performance/" + learned_weights_file + "_ACTIVATIONS.mat"

k_test = 10
################################
###   GET  TRAINED WEIGHTS   ###
################################
if include_DF_HeurNet or include_DF_MST:
    pretrained_conv1d_1_kernel_DF = tw['DF_conv1d_1_kernel']
    pretrained_conv1d_1_bias_DF =tw['DF_conv1d_1_bias']
    pretrained_conv1d_2_kernel_DF =tw['DF_conv1d_2_kernel']
    pretrained_conv1d_2_bias_DF =tw['DF_conv1d_2_bias']
if include_DF_MST:
    pretrained_conv1d_3_kernel_DF =tw['DF_conv1d_3_kernel']
    pretrained_conv1d_3_bias_DF =tw['DF_conv1d_3_bias']
pretrained_conv2d_1_kernel_PZ = tw['PZ_conv2d_1_kernel']
pretrained_conv2d_1_bias_PZ   = tw['PZ_conv2d_1_bias']
pretrained_conv2d_2_kernel_PZ = tw['PZ_conv2d_2_kernel']
pretrained_conv2d_2_bias_PZ =   tw['PZ_conv2d_2_bias']
pretrained_dense_1_kernel_PZ =  tw['dense_1_kernel']
pretrained_dense_1_bias_PZ =    tw['dense_1_bias']
pretrained_dense_2_kernel_PZ =  tw['dense_2_kernel']
pretrained_dense_2_bias_PZ =    tw['dense_2_bias']
pretrained_output_kernel_PZ =   tw['output_kernel']
pretrained_output_bias_PZ =     tw['output_bias']


##############################
###   NETWORK PARAMETERS   ###
##############################
nchannels = 1  # ??? Maybe two because stereo audiofiles?

###GENERAL VARIABLES
# Input data size
length = 20992  # k0

###DEEP_FOURIER LAYERS HYPERPARAMETERS
if include_DF_HeurNet:
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

if include_DF_MST:
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
image_shape = [bands,frames,n_channels]

#INPUT PLACEHOLDER
if include_DF_HeurNet or include_DF_MST:
    x_pl_1 = tf.placeholder(tf.float32, [None, length, nchannels], name='xPlaceholder_1')
else:
    x_pl_1 = tf.placeholder(tf.float32, [None, bands, frames, nchannels], name='xPlaceholder_1')

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

###OUTPUT PLACEHOLDER VARIABLES
y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
y_pl = tf.cast(y_pl, tf.float32)

##############################
###   NETWORK DEFINITION   ###
##############################

print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')

if include_DF_HeurNet:
    ### DEEP FOURIER NETWORK
    with tf.variable_scope('DF_convLayer1'):
        ### INPUT DATA
        print("--- Deep Fourier conv layer 1")
        print('x_pl_1 \t\t', x_pl_1.get_shape()) 
        
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
                                 kernel_initializer=tf.constant_initializer(pretrained_conv1d_1_kernel_DF),
                                 bias_initializer=tf.constant_initializer(pretrained_conv1d_1_bias_DF),
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
        pool1 = MaxPooling1D(    pool_size=DF_pool_size_1,
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
                                 kernel_initializer=tf.constant_initializer(pretrained_conv1d_2_kernel_DF),
                                 bias_initializer=tf.constant_initializer(pretrained_conv1d_2_bias_DF),
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
        pool2 = MaxPooling1D(    pool_size=DF_pool_size_2,
                                        strides=DF_strides_pool2,
                                        padding=DF_padding_pool2, 
                                        name='DF_pool_2'
                                    )
                                       
        # Activation pass, pooling
        a2 = (pool2(z2)) 
        # Reshaping to swtich dimension and get them right (to 41x60 to 60x41x1)
        a2 = tf.transpose(a2, perm=[0,2,1]) 
        a2 = tf.expand_dims(a2, axis=3)
        #a2 = tf.reshape(a2, )
        print('DF_pool2 \t\t', a2.get_shape())
elif include_DF_MST:
    with tf.variable_scope('DF_convLayer1'):
        ### INPUT DATA
        print("--- Deep Fourier conv layer 1")
        print('x_pl_1 \t\t', x_pl_1.get_shape())
        ### CONV1 LAYER
        z1 = tf.layers.conv1d(inputs=x_pl_1,
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
        print('DF_conv1 \t\t', z1.get_shape())

    with tf.variable_scope('DF_convLayer2'):
        ### CONV2 LAYER
        print("--- Deep Fourier conv layer 2")
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
        # Input pass, activation
        print('DF_conv2 \t\t', z2.get_shape())

    with tf.variable_scope('DF_convLayer3'):
        ### CONV3 LAYER
        print("--- Deep Fourier conv layer 3")
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
        # Input pass, activation
        # Reshaping to swtich dimension and get them right (to 41x60 to 60x41x1)
        z3 = tf.transpose(z3, perm=[0, 2, 1])
        a2 = tf.expand_dims(z3, axis=3)
        print('DF_conv3 \t\t', a2.get_shape())
else: #test error of standalone Piczak
    a2=x_pl_1

### PICZAK NETWORK
# Convolutional layers
with tf.variable_scope('PZ_convLayer1'):
    print("--- Piczak")
    conv1 = tf.layers.conv2d(
        kernel_initializer=tf.constant_initializer(pretrained_conv2d_1_kernel_PZ),
        bias_initializer=tf.constant_initializer(pretrained_conv2d_1_bias_PZ),
        inputs=a2, ### NB!!! This is the output of the Deep_Fourier Network
        filters=n_filter_1,
        kernel_size=kernel_size_1,
        strides=kernel_strides_1,
        padding=padding_1,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_1))
    x=conv1 
    print('PZ_conv1 \t\t', x.get_shape())
    pool1 = max_pool2d(x, kernel_size=pool_size_1,stride=pool_strides_1, padding=padding_1)
    x = pool1
    print('PZ_pool1 \t\t', x.get_shape())
    x = tf.nn.dropout(x,dropout_1)

with tf.variable_scope('PZ_convLayer2'):
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
        kernel_initializer=tf.constant_initializer(pretrained_dense_1_kernel_PZ),
        bias_initializer=tf.constant_initializer(pretrained_dense_1_bias_PZ),
        inputs=x,
        units=num_units_3,
        activation=tf.nn.relu,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_3),
    )
    x = dense3
    print('PZ_dense3 \t\t', x.get_shape())
    x = tf.nn.dropout(x,dropout_3)

with tf.variable_scope('PZ_denseLayer4'):
    dense4 = tf.layers.dense(
        kernel_initializer=tf.constant_initializer(pretrained_dense_2_kernel_PZ),
        bias_initializer=tf.constant_initializer(pretrained_dense_2_bias_PZ),
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
        kernel_initializer=tf.constant_initializer(pretrained_output_kernel_PZ),
        bias_initializer=tf.constant_initializer(pretrained_output_bias_PZ),
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

####################################################
###   LOSS, TRAINING AND PERFORMANCE DEFINITION  ###
####################################################

with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), reduction_indices=[1])
    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)

with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))
    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
##############################
###    TESTING on fold 10  ###
##############################

if not ONLY_EXAMPLE_OBSERVATIONS:
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    if include_DF_HeurNet or include_DF_MST:
        test_data=scipy.io.loadmat(data_folder + 'fold{}_wav.mat'.format(k_test))
        test_data=np.expand_dims(test_data['ob_wav'],axis=-1)
    else:
        test_data=scipy.io.loadmat(data_folder + 'fold{}_spcgm.mat'.format(k_test))
        test_data=np.expand_dims(test_data['ob_spcgm'],axis=-1)
    y_test_true=scipy.io.loadmat(data_folder + 'fold{}_labels.mat'.format(k_test))
    y_test_true=utils.onehot(np.transpose(y_test_true['lb']), num_classes) #One-hot encoding labels
else: #calculate only a dummy test error on random data
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#    just for testing
    small_data = 2
    test_data = np.random.normal(0, 1, [small_data, 20992, 1]).astype('float32')
    y_test_true = utils.onehot(np.random.randint(0, 10, small_data), 10)


eat_this = {x_pl_1: test_data, y_pl: y_test_true}
fetches_test = [y, cross_entropy, accuracy]

A = np.sin(2*np.pi*np.arange(20992)*400/20000) #about 400 Hz
B = np.sin(2*np.pi*np.arange(20992)*200/20000) #about 400 Hz
C = A + B
D = np.hstack((np.sin(2*np.pi*np.arange(10992)*200/20000), np.sin(2*np.pi*np.arange(10000)*400/20000)))
D[:3500] = 0
D[14500:18000] = 0
E = np.sin(2*np.pi*np.arange(20992)*20/20000*150/20992*np.arange(1, 20993))

#wav_signals = np.vstack((A, B, C, D, E))
wav_signals = scipy.io.loadmat("data/fold10_RANDOM_OBS")['picked_obs']

all_spcgm = np.zeros((0, 60, 41), np.float64)
for j in range(wav_signals.shape[0]):
    spcgm = librosa.feature.melspectrogram(wav_signals[j], hop_length=512, n_fft=1024, sr=22050, n_mels=60) #how Karol does it
    spcgm = spcgm[:, :41] #for some reason the spcgm returned by spcgm has a width of 42 so we have to trim it to 41
    spcgm = librosa.logamplitude(spcgm) #how Karol does it
    all_spcgm = np.vstack((all_spcgm, [spcgm]))

wav_signals = np.expand_dims(wav_signals, -1)
synth = {x_pl_1: wav_signals}

sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
sess.run(tf.global_variables_initializer())
if STORE_TEST_ERROR: res = sess.run(fetches=fetches_test, feed_dict=eat_this)
activations_of_wav_signals = sess.run(a2, synth)
sess.close()

if STORE_TEST_ERROR:
    y_test_pred = res[0]
    test_loss = res[1]
    test_accuracy = res[2]
    #from softmax to class
    y_test_pred = np.argmax(y_test_pred, axis=1) 
    y_test_true = utils.onehot_inverse(y_test_true)
    conf_mat = confusion_matrix(y_test_true, y_test_pred, labels=range(10))
    cba = utils.classbal_acc(conf_mat)
    
    mdict = {'test_loss': test_loss, 'test_accuracy': test_accuracy, 'conf_mat': conf_mat, 'acc_classbal': cba}
    scipy.io.savemat(save_path_perf, mdict)
    print("performance saved under %s...." % save_path_perf)

if STORE_ACTIVATIONS: scipy.io.savemat(save_path_activ, {'wavact' : activations_of_wav_signals, 'wav': wav_signals, 'mel': all_spcgm})

print("<><><><><><><><> the entire program finished without errors!! <><><><><><><><>")
