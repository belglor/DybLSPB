% clear all; close all; clc;
% 
%========================================
folder = 'results_mat/performance/';
plot_folder = 'accuracy_plots/';
loadfile = 'piczak_A_unbal_LR0-01_ME300_ACCURACY.mat'; 
%========================================

load([folder loadfile]);
max_epochs=300;
%To resize to Matlab conventions
best_epoch=best_epoch+1;
best_bal_epoch=best_bal_epoch+1;

figure(2)
plot(1:max_epochs,train_accuracy)
hold on;
plot(1:max_epochs,valid_accuracy)
plot(1:max_epochs,bal_valid_accuracy)
plot(best_bal_epoch,best_bal_valid_accuracy,'or','MarkerSize',10);
text(double(best_bal_epoch),double(best_bal_valid_accuracy+0.02),num2str(best_bal_valid_accuracy),'fontSize',10,'color','red');
xlabel('Epoch','FontSize',20)
lgd=legend('Training accuracy','Validation accuracy','Balanced validaction accuracy');
lgd.FontSize = 15;
lgd.Location = 'best';
saveas(gcf, [folder plot_folder loadfile], 'png')
saveas(gcf, [folder plot_folder loadfile], 'fig')