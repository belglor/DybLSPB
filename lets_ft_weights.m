%===============================
weight_file = 'results_mat/trainedweights/LarsDeepFourier_dfT_dfcnn1X_dfcnn2X_dfcnn3X_pzcnn1TL_pzcnn2NL_pzfc1NL_pzfc2NL_pzoutL_A_unbal_LR0-005_ME300_LRD_BAL_WEIGHTS';
N = 1024;
num_filters = 512;
%===============================
close all
sampl_freq_Hz = 22050;
load(weight_file);

filters1 = DF_conv1d_1_kernel;

% DF first conv layer filters
inspected_f1 = 1:5;
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
FILTERS1 = squeeze(fft(filters1)); % this is the column-wise fft! % For matrices, the fft operation is applied to each column.
FILTERS1 = FILTERS1(first_half, :);
% FILTERS1 = [1 0 10 2; 3 2 1 0; 8 2 4 9; 7 1 2 1; 0 6 1 1]; % for testing
freq_axis = 1;
filter_idx_axis = 2;
figure(1337)
subplot(2, 1, 1)
imagesc(abs(FILTERS1).^2)
title('FFTs of 1st DF layer kernels')
xlabel('filter index')
ylabel('spectrum (0 to half sampl. freq.)')
colorbar

[~, midfreq] = max(FILTERS1, [], freq_axis);
[mf_sorted, I] = sort(midfreq);
FILTERS1_sort = FILTERS1(:, I);
subplot(2, 1, 2)
imagesc(abs(FILTERS1_sort).^2)
title('FFTs of 1st DF layer kernels')
xlabel('filter index (sorted)')
ylabel('spectrum (0 to half sampl. freq.)')
colorbar
saveas(gcf, weight_file, 'png')
saveas(gcf, weight_file, 'fig')
