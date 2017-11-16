#======= ADJUSTABLE PARAMETERS ================================================
num_total_folds = 10
whereAreTheFiles = "UrbanSound8K/audio/" #directory that contains the FOLDERS "fold1", "fold2" ... "fold10"
where2SaveIt = "" # "" to save it in the same location as the code, otherwise "path/to/folder/"
silence_threshold = -70.0 #the value that Karol uses
name_verite = ""
#==============================================================================

import numpy as np
import librosa
#import matplotlib.pyplot as plt
import os
import scipy.io
from parse import parse

FRAMES_PER_SEGMENT = 41  # 41 frames ~= 950 ms segment length @ 22050 Hz
TIME_WINDOW_SIZE = 512 * FRAMES_PER_SEGMENT   # 23 ms per frame @ 22050 Hz #gives the 950ms from Piczak 
STEP_SIZE = 512 * FRAMES_PER_SEGMENT // 2

sampl_freq_Hz = 22050
num_classes = 10
img_height = 60
img_width = 41 
fft_window_len = 1024
N_vec = np.zeros(num_total_folds, np.float64)
N2_vec = np.zeros(num_total_folds, np.float64)
tooShortList_mat = np.zeros((num_total_folds, num_classes), np.float64)
classPriorsRaw_mat = np.zeros((num_total_folds, num_classes), np.float64)
classPriorsBeforeWindowing_mat = np.zeros((num_total_folds, num_classes), np.float64)
classPriorsAfterWindowing_mat = np.zeros((num_total_folds, num_classes), np.float64)

for fold_num in np.arange(1,num_total_folds+1):
    print("")
    print("-----------------------------------------------------------")
    print("------------- FOLD %d -------------------------------------" %fold_num)
    print("-----------------------------------------------------------")
    direc = whereAreTheFiles + "fold" + str(fold_num)
    audiofiles = os.listdir(direc)
    audiofiles.remove(".DS_Store")
    too_quiet_ctr = 0
    N =len(audiofiles)
    N_vec[fold_num-1] = N
    clip_len = np.zeros(N, np.float64) #get the lengths of the respective audio sequences
    observations_wav = np.zeros((0, TIME_WINDOW_SIZE), np.float64)
    observations_spcgm = np.zeros((0, img_height, img_width), np.float64)
    labels = np.zeros(0, int)
    num_too_short = 0 #number of audioclips that are too short to give even one observation
    classPriorsRaw = np.zeros(num_classes, np.float64)
    classPriorsBeforeWindowing = np.zeros(num_classes, np.float64)
    classPriorsAfterWindowing = np.zeros(num_classes, np.float64)
    
    for n, au in enumerate(audiofiles):
        tmp = parse("{}-{}-{}-{}.wav", au)
        label_for_file = int(tmp[1])
        classPriorsRaw[label_for_file] += 1
        print("processing soundfile {0} of {1}, label {3}, named {2}".format(n, N, au, label_for_file))
        audioclip, _ = librosa.load(direc + "/" + au, sr=sampl_freq_Hz)
        clip_len[n] = len(audioclip)
        classPriorsBeforeWindowing[label_for_file] += 1
        normalization_factor = 1.0 / np.max(np.abs(audioclip)) #how Karol does it
        audioclip *= normalization_factor #how Karol does it
        s = 0
        while True:
            window_wav = audioclip[(s * STEP_SIZE):(s * STEP_SIZE + TIME_WINDOW_SIZE)] #how Karol does it
            s+=1            
            if len(window_wav) < TIME_WINDOW_SIZE: break
            window_spcgm = librosa.feature.melspectrogram(window_wav, hop_length=512, n_fft=fft_window_len, sr=sampl_freq_Hz, n_mels=img_height) #how Karol does it
            window_spcgm = window_spcgm[:, :img_width] #for some reason the window_spcgm returned by window_spcgm has a width of 42 so we have to trim it to 41
            window_spcgm = librosa.logamplitude(window_spcgm) #how Karol does it
            if np.mean(window_spcgm) <= silence_threshold: #That's what Karol said
                too_quiet_ctr += 1
            else:
                observations_spcgm = np.vstack((observations_spcgm, [window_spcgm]))
                observations_wav = np.vstack((observations_wav, window_wav))
                labels = np.hstack((labels, label_for_file)) #*np.ones(1, int)
                classPriorsAfterWindowing[label_for_file] += 1
    
    tooShortList = classPriorsRaw - classPriorsBeforeWindowing
    tooShortList_mat[fold_num-1] = tooShortList
    classPriorsRaw /= N
    classPriorsRaw_mat[fold_num-1] = classPriorsRaw
    classPriorsBeforeWindowing /= N - np.sum(tooShortList)
    classPriorsBeforeWindowing_mat[fold_num-1] = classPriorsBeforeWindowing
    N2 = np.shape(observations_wav)[0]
    N2_vec[fold_num-1] = N2
    classPriorsAfterWindowing /= N2
    classPriorsAfterWindowing_mat[fold_num-1] = classPriorsAfterWindowing
    print("augmented from {0} to {1} observations.".format(N, N2) )
    print("{0} observations (after windowing) were silent and therefore discarded.".format(too_quiet_ctr))
    name = where2SaveIt + "fold%d"%fold_num    
    scipy.io.savemat(name + name_verite + "_spcgm.mat", dict(ob_spcgm = observations_spcgm) )
    scipy.io.savemat(name + name_verite + "_wav.mat", dict(ob_wav = observations_wav) )
    scipy.io.savemat(name + name_verite + "_labels.mat", dict(lb = labels) )
    scipy.io.savemat(name + name_verite + "_stats.mat", 
     dict(clipLen = clip_len, N=N, N2=N2, tooShortList = tooShortList, classPriorsRaw =classPriorsRaw, 
          classPriorsBeforeWindowing = classPriorsBeforeWindowing,
          classPriorsAfterWindowing = classPriorsAfterWindowing)  )
    print("observations and labels were saved in")
    print(name + name_verite + "_spcgm.mat")
    print(name + name_verite + "_wav.mat")
    print(name + name_verite + "_labels.mat")

scipy.io.savemat(where2SaveIt + "allfolds" + name_verite + "_stats.mat", 
 dict(N=N_vec, N2=N2_vec, tooShortList = tooShortList_mat, classPriorsRaw =classPriorsRaw_mat, 
      classPriorsBeforeWindowing = classPriorsBeforeWindowing_mat,
      classPriorsAfterWindowing = classPriorsAfterWindowing_mat)  )
