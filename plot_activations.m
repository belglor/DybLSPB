load('WITH_ACTIVATION_WITH_PURE_DF_PRETRAINING_deepFourier_dfN_dfcnn1L_dfcnn2L_pzcnn1TX_pzcnn2TX_pzfc1TX_pzfc2TX_pzoutX_A_unbal_LR0-002_ME300_BAL_TESTERROR.mat');
load('/home/lorenzo/Documents/DybLSPB-master/data/fold10_spcgm')
obs_toactivate = [4 5 6 7];

for i = 1:length(obs_toactivate)
    figure
    colormap('jet')
   %Plot a figure for each observation's activation
   subplot(1,2,1)
   colormap('jet')
   image_toplot = reshape(activation_DF2(obs_toactivate(i),:,:), [60 41]);
   image(image_toplot);
   titlestr = "Activation through DF of obsevation number " + obs_toactivate(i);
   title(titlestr)
   
   %Plot the corresponding spectrogram
   subplot(1,2,2)
   colormap('jet')
   image_toplot = reshape(ob_spcgm(obs_toactivate(i),:,:), [60 41]);
   image(image_toplot);
   titlestr = "MEL spectrogram of obsevation number " + obs_toactivate(i);
   title(titlestr)
end