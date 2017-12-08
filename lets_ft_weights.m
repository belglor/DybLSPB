load('results_mat/trainedweights/deepFourier_dfT_dfcnn1X_dfcnn2X_pzcnn1TL_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-002_ME300_BAL_WEIGHTS');
%%
filters1 = reshape(DF_conv2d_1_kernel, [155,80]);
filters2 = reshape(DF_conv2d_2_kernel, [15,80,60]);

sampl_freq_Hz = 22050;

% DF first conv layer filters
inspected_f1 = [1 2];
N = 155;
first_half = 1:round(N/2);
time_axis1 = (1:N)/sampl_freq_Hz;
freq_axis1 = first_half/N*sampl_freq_Hz;
for i = 1:length(inspected_f1)
    figure;
    plot(time_axis1,filters1(:,inspected_f1(i)), 'b.-');
    title_str = "Time-domain representation of kernel response for filter " + num2str(inspected_f1(i));
    title(title_str);
    xlabel 'sec'
    %ylabel 'amplitude'
    
    figure;
    fourier_t1 = fft(filters1(:,inspected_f1(i)));
    fourier_t1 = fourier_t1(first_half); %in an FFT, we always throw away the second half of the sprectrum, since it is the mirror image of the first half
    plot(freq_axis1,abs(fourier_t1).^2, 'b.-');
    title_str = "Frequency-domain representation of kernel response for filter " + num2str(inspected_f1(i));
    title(title_str);
    xlabel 'Hz'
    ylabel 'power'
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