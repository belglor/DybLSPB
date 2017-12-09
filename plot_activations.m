% Load activations
load('./results_mat/performance/WITH_ACTIVATION_WITH_PURE_DF_PRETRAINING_deepFourier_dfN_dfcnn1L_dfcnn2L_pzcnn1TX_pzcnn2TX_pzfc1TX_pzfc2TX_pzoutX_A_unbal_LR0-002_ME300_BAL_TESTERROR.mat');
Ole1_activations = activation_DF2;

load('./results_mat/performance/WITH_ACTIVATION_deepFourier_dfT_dfcnn1X_dfcnn2X_pzcnn1TL_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-002_ME300_BAL_TESTERROR');
Lars1_activations = activation_DF2;

% Load spectrograms
load('./data/fold10_spcgm')
MEL_spectrograms = ob_spcgm;

% Load data labels
load('./data/fold10_labels')
data_labels = lb;

% Clear unused variables
clear acc_classbal activation_DF2 conf_mat lb ob_spcgm test_accuracy test_loss
%% Extract one example per class
labels = unique(data_labels);

%Create class masks
masks = struct;
for i = 1:length(labels)
    key = "mask"+i;
    masks.(key) = find(data_labels == labels(i));
end

%Consider one observation per class
Lars1_obs = [];
Ole1_obs = [];
MEL_obs = [];
for i = 1:length(labels)
    key = "mask"+i;
    mask = masks.(key);
    pick_obs = randi(length(mask),1);
    tmp = Lars1_activations(mask,:,:);
    Lars1_obs(i,:,:) = tmp(pick_obs,:,:);
    tmp = Ole1_activations(mask,:,:);
    Ole1_obs(i,:,:) = tmp(pick_obs,:,:);
    tmp = MEL_spectrograms(mask,:,:);
    MEL_obs(i,:,:) =  tmp(pick_obs,:,:);
    picked(i) = pick_obs;
end

%%
for i = 1:length(labels)
    figure
    
    %Plot a figure for each observation's activation for Ole1
    subplot(1,3,1)
    colormap('jet')
    image_toplot = reshape(Lars1_obs(i,:,:), [60 41]);
    image(image_toplot);
    titlestr = "LARS1: Activation of obs " + picked(i) +". C" + i;
    title(titlestr)
    
    %Plot a figure for each observation's activation for Ole1
    subplot(1,3,2)
    colormap('jet')
    image_toplot = reshape(Ole1_obs(i,:,:), [60 41]);
    image(image_toplot);
    titlestr = "OLE1: Activation of obs " + picked(i) +". C" + i;
    title(titlestr)
    
    %Plot the corresponding spectrogram
    subplot(1,3,3)
    colormap('jet')
    image_toplot = reshape(MEL_obs(i,:,:), [60 41]);
    image(image_toplot);
    titlestr = "MEL of obs " + picked(i) +". C" + i;
    title(titlestr)
end