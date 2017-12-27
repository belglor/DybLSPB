clear all; close all; clc;

load('piczak_A_unbal_LR0-002_ME300_TESTERROR.mat');
pz_TA=test_accuracy;
pz_BTA=acc_classbal;
pz_CM=conf_mat;

load('deepFourier_dfT_dfcnn1X_dfcnn2X_pzcnn1TX_pzcnn2TX_pzfc1TX_pzfc2TX_pzoutX_A_unbal_LR0-002_ME300_TESTERROR.mat');
heuri_all_TA=test_accuracy;
heuri_all_BTA=acc_classbal;
heuri_all_CM=conf_mat;

load('Heuri1_OLA_A_LR0-002_ME100_TESTERROR.mat');
heuri_TA=test_accuracy;
heuri_BTA=acc_classbal;
heuri_CM=conf_mat;

load('MST_OLA_A_LR0-005_GC_ME100_TESTERROR.mat');
mst_TA=test_accuracy;
mst_BTA=acc_classbal;
mst_CM=conf_mat;

load('MST_DF_All_LR0-005_ME300_GC_BAL_TESTERROR.mat');
mst_all_TA=test_accuracy;
mst_all_BTA=acc_classbal;
mst_all_CM=conf_mat;

%% Barplots

% models = categorical({'Piczak','DF all','DF Lars1','DF Lars2','DF Ole1','DF Ole2'});
% TA = [pz_TA df_all_TA df_lars1_TA df_lars2_TA df_ole1_TA df_ole2_TA];
% bar(models,TA)

models ={'HeuriNet (icebreaker)','HeuriNet (random init)','','MSTNet (icebreaker)','MSTNet (random init)','','PiczakNet'};
figure(2)
x=[7 1 2 4 5];
BTA = [pz_BTA heuri_BTA heuri_all_BTA mst_BTA mst_all_BTA];
hold on;
bar(7,pz_BTA,'FaceColor',[1 0.5 0]);
bar([1 2], [heuri_BTA heuri_all_BTA],'blue');
bar([4 5], [mst_BTA mst_all_BTA],'red');

ylabel('Balanced test accuracy','FontSize',15)
% for k = 1:size(BTA,2)
%     b(k).CData = k;
% end
%title ('Balanced test accuracy for trained models')
ylim([0 1]);
set(gcf,'color','white')
set(gca, 'XTick', 1:7,'xTickLabel',models,'fontSize',15)
ax = gca;
ax.XTickLabelRotation = 45;
%b(1).Color = 'red'