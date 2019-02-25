# DEEPFOURIER: CLASSIFYING SOUND WITH CONVOLUTIONAL NEURAL NETWORKS

Link to the final article: https://github.com/belglor/DybLSPB/blob/master/DeepFourier.pdf

**Abstract** In this project we looked into sound classification using convolutional neural networks based on previous work done by Karol J. Piczak [1], who trained his network from Melspectrograms. Our aim was to improve upon the original work by making the classifier end-to-end through integration of the raw audio transformation in the deep learning network itself (which we call DeepFourier). We created 2 new networks, both performing as well as Piczak’s original network. After reading this report, please open DEMO.ipynb to see a quick introduction to our code. 

**Index Terms—** Convolutional Neural Network, sound classification, Mel-spectrogram, Short-Term Fourier Transformation

[1]: Piczak K. J., “Environmental sound classification with convolutional neural networks,” in Machine Learning for Signal Processing (MLSP): 25th International Workshop on Machine Learning for Signal Processing., Boston, USA, IEEE. Sep. 2015.



## Demo code, trainable model and examples

A fully runnable example can be found in the DEMO folder. The file is a jupyter notebook called DEMO.ipynb.

One can find other runnable examples in the main folder, i.e. DF_And_PZ_IceBreaker.py. In order to run them, please check results_mat/trainedweights/missing.txt for a list of 2 weight files you need to download.

Then you can specify in the file DF_And_PZ_IceBreaker.py:
* The network architecture in line 31
* The IceBreaker phase in line 32 (start with Phase 1)

and run it.

Beware that you are not really training the network here, since the ```RUN_FAST = True``` option makes sure you train just 2 epochs on random data. If you want to train on the actual data set, contact us. We did not include the data set here due to the file size.
