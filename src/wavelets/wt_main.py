import time

import numpy as np
import pywt
import torch

from src.functions import create_dataloader, generate_mt_freq, visualize_mt, calculate_corr
from src.params import path_to_serialized, device, Sensors, Imagery, wavelet
from src.wavelets.wt import cwt1d, generate_int_psi_scales, wt_str84wd, wt


def main():
    batch_size = 4
    data_test = np.load('../' + path_to_serialized + 'data_test_small.npy')
    markers_test = np.load('../' + path_to_serialized + 'markers_test_small.npy')
    test_loader = create_dataloader(data_test, markers_test, batch_size)
    eeg, markers = next(iter(test_loader))
    eeg = eeg.to(device)

    frequencies = generate_mt_freq(80, 1, 80, 2)
    sampling_rate = 200
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)

    result = cwt1d(eeg, scales, int_psi_scales=generate_int_psi_scales(scales, wavelet, device), out_dtype='complex')
    # result = get_representation(result)
    for i in range(batch_size):
        visualize_mt(result[i][Sensors.C3.value], frequencies, title=Imagery(torch.argmax(markers[i]).item()))


def speed_test():
    batch_size = 128
    data_test = np.load('../' + path_to_serialized + 'data_test_small.npy')
    markers_test = np.load('../' + path_to_serialized + 'markers_test_small.npy')
    test_loader = create_dataloader(data_test, markers_test, batch_size)
    eeg, markers = next(iter(test_loader))
    eeg = eeg.to(device)

    frequencies = generate_mt_freq(80, 1, 80, 2)
    sampling_rate = 200
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)

    start_time = time.time()  # время начала выполнения
    result = cwt1d(eeg, scales, int_psi_scales=generate_int_psi_scales(scales, wavelet, device), out_dtype='real')
    end_time = time.time()
    execution_time = end_time - start_time
    print('Execution time of wavelet transform in pytorch', execution_time)
    # result = get_representation(result)
    visualize_mt(result[0][Sensors.C3.value], frequencies, title=Imagery(torch.argmax(markers[0]).item()))

    start_time = time.time()
    result, _ = pywt.cwt(eeg.to('cpu').numpy(), scales, wavelet)
    end_time = time.time()
    execution_time = end_time - start_time
    print('Execution time of wavelet transform in numpy', execution_time)
    wt_result = np.transpose(result, (1, 2, 0, 3))
    wt_result = torch.from_numpy(wt_result).float()
    visualize_mt(wt_result[0][Sensors.C3.value], frequencies, title=Imagery(torch.argmax(markers[0]).item()))

    print('sula')


def hand_made_test():
    data_test = np.load('../' + path_to_serialized + 'data_test_small.npy')
    signal = data_test[111][6]

    frequencies = generate_mt_freq(80, 1, 80, 2)
    sampling_rate = 200
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)

    start_time = time.time()
    wt_result = np.real(wt(signal, scales / 10))
    end_time = time.time()
    print('Time fourier', end_time - start_time)
    visualize_mt(wt_result, frequencies)

    start_time = time.time()
    wt_dummy = wt_str84wd(signal, wavelet, scales / 10)
    end_time = time.time()
    print('Time dummy', end_time - start_time)
    visualize_mt(wt_dummy, frequencies)

    start_time = time.time()
    wt_pywt, _ = pywt.cwt(signal, scales, wavelet, method='fft')
    end_time = time.time()
    print('library time', end_time - start_time)
    visualize_mt(wt_pywt, frequencies)


def corr_test():
    data_test = np.load('../' + path_to_serialized + 'data_test_small.npy')

    signal_num = 100
    signals_c3 = data_test[0:signal_num, Sensors.C3.value]
    signals_c4 = data_test[0:signal_num, Sensors.C4.value]

    sum_corr = 0
    for i in range(signal_num):
        sum_corr += calculate_corr(signals_c3[i], signals_c4[i])
    mean_corr = sum_corr / signal_num
    print('Mean corr for signals c3 and c4 electrodes', mean_corr)

    frequencies = generate_mt_freq(80, 1, 80, 2)
    sampling_rate = 200
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    wt_signals_c3, _ = pywt.cwt(signals_c3, scales, wavelet, method='fft')
    wt_signals_c4, _ = pywt.cwt(signals_c4, scales, wavelet, method='fft')

    wt_signals_c3 = wt_signals_c3.transpose(1, 0, 2)
    wt_signals_c4 = wt_signals_c4.transpose(1, 0, 2)

    sum_corr_wt = 0
    for i in range(signal_num):
        sum_corr_wt += calculate_corr(wt_signals_c3[i], wt_signals_c4[i])
    mean_corr_wt = sum_corr_wt / signal_num
    print('Mean corr for wt of signals c3 and c4 electrodes', mean_corr_wt)


if __name__ == '__main__':
    # hand_made_test()
    corr_test()
