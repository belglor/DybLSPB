WAV = 0;
SPC = 1;
%============================================================
path = 'results_mat/performance/';
name1 = 'Heuri1_OLA_';
name2 = '_LR0-002_ME100_RANDOM_OBS';
name3 = '_ACTIVATIONS';
what2plot = SPC;
%============================================================
close all

load([path name1 'O' name2 name3]);
wav_obs = wav;
MEL_spc = mel;
Ole1_act = wavact;
load([path name1 'L' name2 name3]);
Lars1_act = wavact;
load([path name1 'A' name2 name3]);
ALL_act = wavact;

num_rows = 5;
num_cols = 4;


%%
offset = 0;
figure
for i = 0:(num_rows-1)
    if what2plot == WAV
        %Plot the signl in time domain
        subplot(num_rows,1,i+1)
        plot(reshape(wav_obs(i+offset+1,:,:), [1 20992]));
        axis off
        grid
        titlestr = 'wav';
        ylabel("signal"+(i+1))
        if i == 0
            title(titlestr)
        end
    elseif what2plot == SPC
        %Plot the corresponding mel spectrogram
        subplot(num_rows,num_cols,i*num_cols+1)
        colormap('jet')
        image_toplot = reshape(MEL_spc(i+offset+1,:,:), [60 41]);
        image(image_toplot);
        axis off
        titlestr = 'MEL';
        if i == 0
            title(titlestr)
        end
        
        %Plot a figure for each observation's activation for Ole1
        subplot(num_rows,num_cols,i*num_cols+2)
        colormap('jet')
        image_toplot = reshape(Ole1_act(i+offset+1,:,:), [60 41]);
        image(image_toplot);
        axis off
        titlestr = 'Phase 1';
        if i == 0
            title(titlestr)
        end
        
        %Plot a figure for each observation's activation for Ole1
        subplot(num_rows,num_cols,i*num_cols+3)
        colormap('jet')
        image_toplot = reshape(Lars1_act(i+offset+1,:,:), [60 41]);
        image(image_toplot);
        axis off
        titlestr = 'Phase 2';
        if i == 0
            title(titlestr)
        end
        
        %Plot a figure for each observation's activation for Ole1
        subplot(num_rows,num_cols,i*num_cols+4)
        colormap('jet')
        image_toplot = reshape(ALL_act(i+offset+1,:,:), [60 41]);
        image(image_toplot);
        axis off
        titlestr = 'Phase 3';
        if i == 0
            title(titlestr)
        end
    end
end
if what2plot == WAV
    name = [path name1 name2 '_WAV'];
elseif what2plot == SPC
    name = [path name1 name2 name3];
end

saveas(gcf, name, 'fig')
saveas(gcf, name, 'png')