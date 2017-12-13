%============================================================
path = 'results_mat/trainedweights/';
name1 = 'Heuri1_OLA_';
name2 = '_LR0-002_ME100';
name3 = '_BAL_WEIGHTS';
N = 155;
num_filters = 80;
%============================================================
close all
sampl_freq_Hz = 22050;
first_half = 1:round(N/2);

PHASES = 'OLA';
figure(1337)
for p = 1:3
    load([path name1 PHASES(p) name2 name3]);
    FILTERS1 = squeeze(fft(DF_conv1d_1_kernel)); % this is the column-wise fft! % For matrices, the fft operation is applied to each column.
    FILTERS1 = FILTERS1(first_half, :);
    % FILTERS1 = [1 0 10 2; 3 2 1 0; 8 2 4 9; 7 1 2 1; 0 6 1 1]; % for testing
    freq_axis = 1;
    filter_idx_axis = 2;
    subplot(2, 3, p)
    imagesc(abs(FILTERS1))
    title("phase " + p)
    axis off
    [~, midfreq] = max(FILTERS1, [], freq_axis);
    [mf_sorted, I] = sort(midfreq);
    FILTERS1_sort = FILTERS1(:, I);
    subplot(2, 3, 3+p)
    imagesc(abs(FILTERS1_sort))
    axis off
end

name = [path name1 name2 '_FFT_DF_C1'];

saveas(gcf, name, 'fig')
saveas(gcf, name, 'png')
