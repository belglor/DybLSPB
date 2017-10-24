import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import scipy.io
from parse import *

fold_num = 5 #change this to fit the fold number

direc = "UrbanSound8K/audio/fold" + str(fold_num)
audiofiles = os.listdir(direc)

audiofiles.remove(".DS_Store")

N =len(audiofiles)
img_height = 60
fft_window_len = 1024
segment_len = 41 #in samples
print("length of FFT window (seconds): ", float(fft_window_len)/22050)
print("length of FFT segment (seconds): ", segment_len*float(.5*fft_window_len)/22050)
num_channels = 2 #the spectrogram and the deltas

#thedata = np.zeros((N, num_channels, img_height, img_width), np.float64)
#get the lengths of the respective audio sequences
clip_len = np.zeros(N, np.float64)
labels = np.zeros(0, int)
observations = np.zeros((0, img_height, segment_len), np.float64)

def windowing(spcgm):
    s1 = segment_len // 2
    sp_len = np.shape(spcgm)[1]
    sp_hei = np.shape(spcgm)[0]
    nwins = 2*(sp_len // segment_len)-1
    if sp_len % segment_len >= s1:
        nwins+=1
    theWindows = np.zeros((nwins, sp_hei, segment_len), np.float64)
    for ro in range(nwins):
        if ro%2 == 0:
            startpos = (ro//2)*segment_len 
        else:
            startpos = (ro//2)*segment_len+s1
        mask = np.arange(startpos, startpos+segment_len)
        theWindows[ro] = spcgm[:, mask]
    return theWindows, nwins    

for n, au in enumerate(audiofiles):
    print("processing soundfile {0} of {1}, named {2}".format(n, N, au))
    y, sr = librosa.load(direc + "/" + au)
    clip_len[n] = len(y)        
    S = librosa.feature.melspectrogram(y, hop_length=512, 
                                       n_fft=fft_window_len, sr=sr, n_mels=img_height)
    log_S = librosa.power_to_db(S, ref=np.max)
    if np.shape(log_S)[1] < segment_len:
        continue
    tmp = parse("{}-{}-{}-{}.wav", au)
    label_for_file = int(tmp[1])         
    new_wins, n_new_wins = windowing(log_S)
    print("label: " + tmp[1] + "\tlen (time samples)" + str(len(y)) + "\tnum obs. augm. from it: " + str(n_new_wins))
    observations = np.vstack((observations, new_wins))
    labels = np.hstack((labels, label_for_file*np.ones(n_new_wins, int) ))
    
plt.hist(clip_len)
scipy.io.savemat("UrbanSound8K/fold%d.mat"%fold_num, dict(ob = observations, lb = labels))

#small test
#file_idx = 870
#ob_idx = np.arange(5425, 5432)   
#y, sr = librosa.load(direc + "/" + audiofiles[file_idx])
#S = librosa.feature.melspectrogram(y, hop_length=512, 
#                                   n_fft=fft_window_len, sr=sr, n_mels=img_height)
#log_S = librosa.power_to_db(S, ref=np.max)
#scipy.io.savemat("UrbanSound8K/870.mat", dict(ob = observations[ob_idx], full = log_S))