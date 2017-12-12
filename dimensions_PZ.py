###PICZACK HYPERPARAMETERS
# Bands : related to frequencies. Frames : related to audio sequence. 2 channels (the spec and the delta's)
bands, frames, n_channels = 60, 41, 1
image_shape = [bands, frames, n_channels]

# First convolutional ReLU layer
n_filter_1 = 80
kernel_size_1 = [57, 6]
kernel_strides_1 = (1, 1)
# Activation in the layer definition
# activation_1="relu"
# L2 weight decay
l2_1 = 0.001

# Dropout rate after pooling
dropout_1 = 0.5

# First MaxPool layer
pool_size_1 = (4, 3)
pool_strides_1 = (1, 3)
padding_1 = "valid"

### Second convolutional ReLU layer
n_filter_2 = 80
kernel_size_2 = [1, 3]
kernel_strides_2 = (1, 1)
# Activation in the layer definition
# activation_2="relu"
l2_2 = 0.001

# Scond MaxPool layer
pool_size_2 = (1, 3)
pool_strides_2 = (1, 3)
padding_2 = "valid"

# Third (dense) ReLU layer
num_units_3 = 5000
# Activation in the layer definition
# activation_3 = "relu"
dropout_3 = 0.5
l2_3 = 0.001

# Fourth (dense) ReLU layer
num_units_4 = 5000
# Activation in the layer definition
# activation_4 = "relu"
dropout_4 = 0.5
l2_4 = 0.001

# Output softmax layer (10 classes in UrbanSound8K)
num_classes = 10
# Activation in the layer definition
# activation_output="softmax"
l2_output = 0.001