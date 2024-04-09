import numpy as np
import pywt
import torch

from src.wavelets.dwt1d import cwt1d, generate_int_psi_scales, get_representation
from src.functions import create_dataloader, generate_mt_freq, visualize_mt
from src.params import path_to_serialized, device, Sensors, Imagery, wavelet

if __name__ == '__main__':
    batch_size = 64
    data_test = np.load(path_to_serialized + 'data_test_small.npy')
    markers_test = np.load(path_to_serialized + 'markers_test_small.npy')
    test_loader = create_dataloader(data_test, markers_test, batch_size)
    eeg, markers = next(iter(test_loader))
    eeg = eeg.to(device)

    frequencies = generate_mt_freq(80, 1, 80, 2)
    sampling_rate = 200
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)

    result = cwt1d(eeg, scales, int_psi_scales=generate_int_psi_scales(scales, wavelet, device), out_dtype='complex')
    result = get_representation(result)
    for i in range(batch_size):
        visualize_mt(result[i][Sensors.C3.value], frequencies, title=Imagery(torch.argmax(markers[i]).item()))

    print('sula')
