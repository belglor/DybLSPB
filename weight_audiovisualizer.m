load('C:\Users\Peti\Desktop\Courses\DeepLearning\Project\deepFourier_dfT_dfcnn1X_dfcnn2X_pzcnn1TL_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-002_ME300_WEIGHTS.mat')
%load('C:\Users\Peti\Desktop\Courses\DeepLearning\Project\piczak_A_unbal_LR0-002_ME300_WEIGHTS.mat')
%load('C:\Users\Peti\Desktop\Courses\DeepLearning\Project\deepFourier_dfT_cnn1T_cnn2N_fcN_A_unbal_LR0-002_ME300_BAL_WEIGHTS.mat')

%% Create cell for storing all weights of Piczak 1st layer

weight_cell_1=cell(size(PZ_conv2d_1_kernel,4),1);
for i=1:size(PZ_conv2d_1_kernel,4)
    max_val=max(PZ_conv2d_1_kernel(:,:,1,i));
    weight_cell_1{i}=(PZ_conv2d_1_kernel(:,:,1,i))./max_val;  
end

%% Creating grid of visualized weights
fig2=figure(2)
for i=1:40 %FIRST 40 WEIGHTS
    
    subplot(4,10,i)   
    pic=weight_cell_1{i};
    RI = imref2d(size(weight_cell_1{1}));
    RI.XWorldLimits = [0 1];
    RI.YWorldLimits = [0 2];

    imshow(pic,RI);
    axis off;
end
truesize(fig2,[600 700])

fig3=figure(3)
for i=41:80 % SECOND 40 WEIGHTS
    
    subplot(4,10,i-40)   
    pic=weight_cell_1{i};
    RI = imref2d(size(weight_cell_1{1}));
    RI.XWorldLimits = [0 1];
    RI.YWorldLimits = [0 2];

    imshow(pic,RI);
    axis off;
end
truesize(fig3,[600 700])

%% Checking Deep Fourier first layer weights

for i=1:30 %CHANGE THIS TO 1:80 IF YOU WANT ALL THE WEIGHTS
    figure(i)
    tmp=DF_conv2d_1_kernel(:,:,i);
    plot(tmp)
    drawnow;
    pause(0.5)
end

%% Checking sound of filter

%Choosing one of the filter for playing as audio
tmp2=DF_conv2d_1_kernel(:,:,17);

tmp_sound=[];

%Repeating the filter to create a ~3.5 sec audio file
for i=1:500
   tmp_sound=[tmp_sound;tmp2]; 
end

%% Playing the audio at 22.05kHz
soundsc(tmp_sound,22050);

%% Visualizing spectrogram of kernel weights

tmp_spect=abs(spectrogram(tmp2,77));
figure
imagesc(tmp_spect');