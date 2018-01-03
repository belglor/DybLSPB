# DybLSPB

Welcome to the DeepFourier project. You can run, for example, the file DF_And_PZ_IceBreaker.py.

To do this, please check results_mat/trainedweights/missing.txt for a list of 2 weight files you need to download.

Then you can specify in the file DF_And_PZ_IceBreaker.py:
* The network architecture in line 31
* The IceBreaker phase in line 32 (start with Phase 1)

and run it.

Beware that you are not really training the network here, since the ```RUN_FAST = True``` option makes sure you train just 2 epochs on random data. If you want to train on the actual data set, contact us. We did not include the data set here due to the file size.
