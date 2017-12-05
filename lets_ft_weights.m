load('deepFourier_dfT_dfcnn1X_dfcnn2X_pzcnn1TL_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-002_ME300_BAL_WEIGHTS');
%%
filters1 = reshape(DF_conv2d_1_kernel, [155,80]);
filters2 = reshape(DF_conv2d_2_kernel, [15,80,60]);

% DF first conv layer filters
inspected_f1 = [1 2];
time_axis1 = 1:155;
for i = 1:length(inspected_f1)
    figure;
    plot(time_axis1,filters1(:,inspected_f1(i)));
    title_str = "Time-domain representation of kernel response for filter " + "inspected_f1";
    title(title_str);
    
    figure;
    fourier_t1 = fft(filters1(:,inspected_f1(i)));
    plot(time_axis1,abs(fourier_t1).^2);
    title_str = "Frequency-domain representation of kernel response for filter " + "inspected_f1";
    title(title_str);
end
% % DF second conv layer filters
% inspected_f2 = [1 2];
% for i = 1:length(inspected_f2)
%     figure;
%     to_plot = reshape(filters2(:,:,inspected_f2(i)), [15,80]);
%     image(to_plot)
%     
%     figure;
%     fourier_t2 = fft(filters2(:,:,inspected_f2(i)));
%     fourier_t2 = abs(fourier_t2).^2;
%     to_plot = reshape(fourier_t2, [15,80]);
%     image(to_plot);
% 
% end