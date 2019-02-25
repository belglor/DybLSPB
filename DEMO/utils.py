import numpy as np
    
def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def onehot_inverse(labels_onehot):
    #expected shape: (num_obs, num_classes)
    return np.argmax(labels_onehot, axis=1)
    
def classbal_acc(conf_mat):
    num_obs_per_class = np.sum(conf_mat, axis=1)
    truepos_per_class = np.diag(conf_mat)
    return np.mean(truepos_per_class/num_obs_per_class)