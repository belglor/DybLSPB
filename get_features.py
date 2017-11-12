import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import scipy.io
from parse import *

#======= ADJUSTABLE PARAMETERS ================================================
fold_num = 1 #change this to fit the fold number
whole_audios = True #if True, every audio contributes exactly one observation, therefore the length varies
#==============================================================================


direc = "UrbanSound8K/audio/fold" + str(fold_num)
audiofiles = os.listdir(direc)

audiofiles.remove(".DS_Store")

num_classes = 10
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
if whole_audios:
    labels = np.zeros(N, int)
    observations = np.zeros((N, segment_len), np.float64)
labels = np.zeros(0, int)
observations = np.zeros((0, img_height, segment_len), np.float64)

num_too_short = 0 #number of audioclips that are too short to give even one observation
classPriorsRaw = np.zeros(num_classes, np.float64)
classPriorsBeforeWindowing = np.zeros(num_classes, np.float64)
classPriorsAfterWindowing = np.zeros(num_classes, np.float64)

def windowing(spcgm):
    s1 = segment_len // 2
    sp_len = np.shape(spcgm)[1]
    sp_hei = np.shape(spcgm)[0]
    nwins = 2*(sp_len // segment_len)-1
    irregular_window = True #will happen nearly always
    rest_at_the_end = sp_len % segment_len
    if rest_at_the_end == s1: #everything fits perfectly
        extra_windows = 1
        irregular_window = False
    elif rest_at_the_end == 0: #everything fits perfectly
        extra_windows = 0
        irregular_window = False
    elif rest_at_the_end < s1: extra_windows = 1
    elif rest_at_the_end > s1: extra_windows = 2
    nwins += extra_windows  

    theWindows = np.zeros((nwins, sp_hei, segment_len), np.float64)
    if irregular_window:
        win_range=nwins-1
    else:
        win_range=nwins
    for ro in range(win_range):
        if ro%2 == 0:
            startpos = (ro//2)*segment_len
        else:
            startpos = (ro//2)*segment_len+s1
        mask = np.arange(startpos, startpos+segment_len)
        theWindows[ro] = spcgm[:, mask]

    if irregular_window:
        mask=np.arange(sp_len-segment_len,sp_len)
        theWindows[-1]=spcgm[:,mask]
    return theWindows, nwins

for n, au in enumerate(audiofiles):
    tmp = parse("{}-{}-{}-{}.wav", au)
    label_for_file = int(tmp[1])
    classPriorsRaw[label_for_file] += 1
    print("processing soundfile {0} of {1}, label {3}, named {2}".format(n, N, au, label_for_file))
    y, sr = librosa.load(direc + "/" + au)
    clip_len[n] = len(y)
    S = librosa.feature.melspectrogram(y, hop_length=512,
                                       n_fft=fft_window_len, sr=sr, n_mels=img_height)
    log_S = librosa.power_to_db(S, ref=np.max)
    if np.shape(log_S)[1] < segment_len: continue
    classPriorsBeforeWindowing[label_for_file] += 1
    new_wins, n_new_wins = windowing(log_S)
    classPriorsAfterWindowing[label_for_file] += n_new_wins
#    print("label: " + tmp[1] + "\tlen (time samples)" + str(len(y)) + "\tnum obs. augm. from it: " + str(n_new_wins))
    observations = np.vstack((observations, new_wins))
    labels = np.hstack((labels, label_for_file*np.ones(n_new_wins, int) ))

tooShortList = classPriorsRaw - classPriorsBeforeWindowing
classPriorsRaw /= N
classPriorsBeforeWindowing /= N - np.sum(tooShortList)
N2 = np.shape(observations)[0]
classPriorsAfterWindowing /= N2
print("augmented from {0} to {1} observations.".format(N, N2) )
print("classPriorsBeforeWindowing: ", classPriorsBeforeWindowing)
print("classPriorsAfterWindowing: ", classPriorsAfterWindowing)

plt.hist(clip_len)
scipy.io.savemat("UrbanSound8K/fold%d_with_irregwin.mat"%fold_num, 
                 dict(N=N, N2=N2, ob = observations, 
                      lb = labels, tooShortList = tooShortList, classPriorsRaw =classPriorsRaw, 
                      classPriorsBeforeWindowing = classPriorsBeforeWindowing,
                      classPriorsAfterWindowing = classPriorsAfterWindowing
                      )
                 )

#small test
#file_idx = 870
#ob_idx = np.arange(5425, 5432)
#y, sr = librosa.load(direc + "/" + audiofiles[file_idx])
#S = librosa.feature.melspectrogram(y, hop_length=512,
#                                   n_fft=fft_window_len, sr=sr, n_mels=img_height)
#log_S = librosa.power_to_db(S, ref=np.max)
#scipy.io.savemat("UrbanSound8K/870.mat", dict(ob = observations[ob_idx], full = log_S))
