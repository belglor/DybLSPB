% clear all; close all; clc;
% 
%========================================
folder = 'results_mat/performance/';
plot_folder = 'confusion_matrices/';
loadfile = 'LarsDeepFourier_dfT_dfcnn1X_dfcnn2X_dfcnn3X_pzcnn1NL_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-005_ME300_BAL_TESTERROR'; 
%========================================

load([folder loadfile]);
num_classes=10;

imagesc(conf_mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)
textStrings = num2str(conf_mat(:));  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:num_classes);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(conf_mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:num_classes,...                         %# Change the axes tick marks
        'XTickLabel',{'AI','CA','CH','DO','DR','EN','GU','JA','SI','ST'},...  %#   and tick labels
        'YTick',1:num_classes,...
        'YTickLabel',{'AI','CA','CH','DO','DR','EN','GU','JA','SI','ST'},...
        'TickLength',[0 0]);
title ( ['CBA (test): ' num2str(acc_classbal)] )  
xlabel 'predicted'
ylabel 'actual'
saveas(gcf, [folder plot_folder loadfile], 'png')
saveas(gcf, [folder plot_folder loadfile], 'fig')