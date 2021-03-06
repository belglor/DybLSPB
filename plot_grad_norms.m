close all
clear all
dbstop error

%===================================================================
% specify all this manually and double-check!
loadfile = 'results_mat/performance/LarsDeepFourier_dfT_dfcnn1X_dfcnn2X_dfcnn3X_pzcnn1TL_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-005_ME300_LRD_GRADIENTNORMS';
% load(loadfile)
names_MST_and_PZ = {'MST1\_kernel',...
'MST1\_bias',...
'MST2\_kernel',...
'MST2\_bias',...
'MST3\_kernel',...
'MST3\_bias',...
'PZC1\_kernel',...
'PZC1\_bias',...
'PZC2\_kernel',...
'PZC2\_bias',...
'PZD1\_kernel',...
'PZD1\_bias',...
'PZD2\_kernel',...
'PZD2\_bias',...
'PZO\_kernel',...
'PZO\_bias'};
MST_IDX = 1:6;
PZC_IDX = 7:10;
PZD_IDX = 11:14;
PZO_IDX = 15:16;
names_HeurNet_and_PZ = {'HeurNet1\_kernel',...
'HeurNet1\_bias',...
'HeurNet2\_kernel',...
'HeurNet2\_bias',...
'PZC1\_kernel',...
'PZC1\_bias',...
'PZC2\_kernel',...
'PZC2\_bias',...
'PZD1\_kernel',...
'PZD1\_bias',...
'PZD2\_kernel',...
'PZD2\_bias',...
'PZO\_kernel',...
'PZO\_bias'};
weight_names = names_MST_and_PZ;
fig_title = 'Lars1 MST, LR decay (starting at 0.005)'; %pick something meaningful please
%===================================================================

load(loadfile)
num_it = size(GN_L2, 1);
GN_L2 = GN_L2';
iteration_idx = 1:num_it;
it_per_epoch = num_it/300;
epoch_idx = iteration_idx/it_per_epoch;

figure(64137)
title(fig_title)
subplot(2, 2, 1)
plot(epoch_idx, GN_L2(MST_IDX, :))
% ylabel 'L2 norm'
xlabel 'epoch'
ylabel 'gradient L2 norm'
legend(weight_names{MST_IDX})

subplot(2, 2, 2)
plot(epoch_idx, GN_L2(PZC_IDX, :))
xlabel 'epoch'
legend(weight_names{PZC_IDX})

subplot(2, 2, 3)
plot(epoch_idx, GN_L2(PZD_IDX, :))
xlabel 'epoch'
ylabel 'gradient L2 norm'
legend(weight_names{PZD_IDX})

subplot(2, 2, 4)
plot(epoch_idx, GN_L2(PZO_IDX, :))
xlabel 'epoch'
legend(weight_names{PZO_IDX})

saveas(gcf, loadfile, 'png')
saveas(gcf, loadfile, 'fig')