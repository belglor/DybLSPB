### IMPORT PACKAGES
import scipy

#from __future__ import absolute_import, division, print_function 

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import os
import sys
sys.path.append(os.path.join('.', '..')) 
import utils 
import batch_loader as bl
from scipy import io

### LOAD DATA
data1 = scipy.io.loadmat("fold2.mat");
lb1_vec = np.transpose(data1["lb"]);
ob1 = data1["ob"];
ob1 = np.reshape(ob1, [ob1.shape[0], ob1.shape[1], ob1.shape[2], 1]);

### ONE-OUT-OF-K ENCODING OF LABELS
import sklearn.preprocessing
a = lb1_vec;
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(max(a))+1))
lb1 = label_binarizer.transform(a)
#print('{0}'.format(b))

### MINIBATCH TESTING
subdata = ob1[:41, :, : ,:]
sublabels = lb1[:41, :]
batch_size = 10;
loader = bl.batch_loader(subdata, sublabels, batch_size);

print('Loader variables:')
print('loader.idx')
print(loader.idx)
print('loader.iters_per_epoch')
print(loader.iters_per_epoch)
print('loader.epoch_idx')
print(loader.epoch_idx)

epoch = 0;
max_epochs = 5

while(epoch <= max_epochs):
    batch_data, batch_labels = loader.next_batch();
    print('====================================================')
    print('Current idx in epoch:')
    print(loader.epoch_idx)
    print('Current batch indexes:')
    print(loader.batch_idx)
    if(loader.end_epoch):
        epoch = epoch + 1;
        print('### Finished epoch: ###')
        print(epoch)
    

