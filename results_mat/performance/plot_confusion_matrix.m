load('piczak_CV_unbal_LR0-002_ME300_ACCURACY.mat');

mat=reshape(conf_mat(1,:,:),[10 10]);

num_classes=10;

imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)
textStrings = num2str(mat(:));  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:num_classes);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:num_classes,...                         %# Change the axes tick marks
        'XTickLabel',{'AI','CA','CH','DO','DR','EN','GU','JA','SI','ST'},...  %#   and tick labels
        'YTick',1:num_classes,...
        'YTickLabel',{'AI','CA','CH','DO','DR','EN','GU','JA','SI','ST'},...
        'TickLength',[0 0]);