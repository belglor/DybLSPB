% clear all; close all; clc;
% 
%========================================
folder = '';
plot_folder = 'accuracy_plots/';
loadfile1 = 'Heuri1_OLA_O_LR0-002_ME100_ACCURACY.mat';
loadfile2 = 'Heuri1_OLA_L_LR0-002_ME100_ACCURACY.mat';
loadfile3 = 'Heuri1_OLA_A_LR0-002_ME100_ACCURACY.mat';
%========================================
load([folder loadfile1]);
ph1_train_acc=train_accuracy;
ph1_valid_acc=valid_accuracy;
ph1_bal_valid_acc=bal_valid_accuracy;

load([folder loadfile2]);
ph2_train_acc=train_accuracy;
ph2_valid_acc=valid_accuracy;
ph2_bal_valid_acc=bal_valid_accuracy;

load([folder loadfile3]);
ph3_train_acc=train_accuracy;
ph3_valid_acc=valid_accuracy;
ph3_bal_valid_acc=bal_valid_accuracy;
%%
phase1_epochs=1:100;
phase2_epochs=101:200;
phase3_epochs=201:300;
%To resize to Matlab conventions
% best_epoch=best_epoch+1;
% best_bal_epoch=best_bal_epoch+1;

clf
figure(1)
hold on
% PHASE 1
plot(phase1_epochs,ph1_train_acc,'b-')
plot(phase1_epochs,ph1_valid_acc,'r-')
plot(phase1_epochs,ph1_bal_valid_acc,'-','color',[0.7 0 0.3])

% PHASE 2
plot(phase2_epochs,ph2_train_acc,'b-')
plot(phase2_epochs,ph2_valid_acc,'r-')
plot(phase2_epochs,ph2_bal_valid_acc,'-','color',[0.7 0 0.3])

% PHASE 3
plot(phase3_epochs,ph3_train_acc,'b-')
plot(phase3_epochs,ph3_valid_acc,'r-')
plot(phase3_epochs,ph3_bal_valid_acc,'-','color',[0.7 0 0.3])

% Vertical separator lines
plot([100 100],[0 1],'-','Linewidth',3,'color',[0.3 0.3 0.3])
plot([200 200],[0 1],'-','Linewidth',3,'color',[0.3 0.3 0.3])

text(42,0.9,'Phase 1','fontSize',16,'fontWeight','bold','color','black');
text(142,0.9,'Phase 2','fontSize',16,'fontWeight','bold','color','black');
text(242,0.9,'Phase 3','fontSize',16,'fontWeight','bold','color','black');



% plot(best_bal_epoch,best_bal_valid_accuracy,'or','MarkerSize',10);
% text(double(best_bal_epoch),double(best_bal_valid_accuracy+0.02),num2str(best_bal_valid_accuracy),'fontSize',10,'color','red');
xlabel('Epoch','FontSize',20)
ylabel('Accuracy','FontSize',20)
lgd=legend('Training accuracy','Validation accuracy','Balanced validation accuracy');
lgd.FontSize = 15;
lgd.Location = 'best';

% saveas(gcf, [folder plot_folder loadfile1], 'png')
% saveas(gcf, [folder plot_folder loadfile1], 'fig')