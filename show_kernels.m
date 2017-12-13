%============================================================
path = 'results_mat/trainedweights/';
name1 = 'Heuri1_OLA_';
name2 = '_LR0-002_ME100';
name3 = '_BAL_WEIGHTS';
N = 155;
num_filters = 80;
%============================================================

load([path name1 'O' name2 name3]);

close all
sampl_freq_Hz = 22050;
filters1 = DF_conv1d_1_kernel;

% DF first conv layer filters
inspected_f1 = 1:4;
first_half = 1:round(N/2);
time_axis1 = (1:N)/sampl_freq_Hz;
freq_axis1 = first_half/N*sampl_freq_Hz;
for i = 1:length(inspected_f1)
    figure;
    plot(time_axis1,filters1(:,inspected_f1(i)), 'b.-');
    title_str = "kernel " + num2str(inspected_f1(i));
    title(title_str);
    xlabel 'sec'
    %ylabel 'amplitude'
    figure;
    fourier_t1 = fft(filters1(:,inspected_f1(i)));
    fourier_t1 = fourier_t1(first_half); %in an FFT, we always throw away the second half of the sprectrum, since it is the mirror image of the first half
    plot(freq_axis1,abs(fourier_t1).^2, 'b.-');
    title_str = "FFT of kernel " + num2str(inspected_f1(i));
    title(title_str);
    xlabel 'Hz'
    ylabel 'power'
end
