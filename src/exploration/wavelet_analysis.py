import numpy as np
import pywt
import torch
from torch import tensor
import time


from src.functions import (get_subject_imageries, normalize_data, MorletTransform, visualize_mt, generate_mt_freq,
                           visualize_sample)
from src.params import path_to_serialized, path_to_dataset, Imagery, Sensors, sampling_rate, wavelet, device
from wavelets.dwt1d import generate_int_psi_scales, cwt1d

if __name__ == '__main__':
    # Get one subject and average his left and right hand motor imagery wavelet image to see if there presents
    # some patterns. For example get E subject
    subject = 'E'
    data, markers = get_subject_imageries(subject, '../' + path_to_dataset)
    data = normalize_data(data)

    lh_data = data[markers == Imagery.LEFT_HAND.value]
    rh_data = data[markers == Imagery.RIGHT_HAND.value]
    passive_data = data[markers == Imagery.PASSIVE.value]

    points_num = 80
    frequencies = generate_mt_freq(points_num, bottom=1, top=80, power=2)
    ###
    # i = 0
    # while True:
    #     i += 1
    #     lh_trial = lh_data[12 + i]
    #     rh_trial = rh_data[12 + i]
    #     lh_trial_magn = MorletTransform(lh_trial)
    #     rh_trial_magn = MorletTransform(rh_trial)
    #
    #     visualize_mt(lh_trial_magn[Sensors.C3.value], frequencies, title='Left hand mt C3')
    #     visualize_mt(rh_trial_magn[Sensors.C3.value], frequencies, title='Right hand mt C3')
    #
    #     visualize_mt(lh_trial_magn[Sensors.Cz.value], frequencies, title='Left hand mt Cz')
    #     visualize_mt(rh_trial_magn[Sensors.Cz.value], frequencies, title='Right hand mt Cz')
    #
    #     visualize_mt(lh_trial_magn[Sensors.C4.value], frequencies, title='Left hand mt C4')
    #     visualize_mt(rh_trial_magn[Sensors.C4.value], frequencies, title='Right hand mt C4')
    # exit()
    ###

    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    random_number = 12

    start_time = time.time()
    mt_lh = cwt1d(tensor(lh_data).to(device), scales, int_psi_scales=int_psi_scales, out_dtype='complex')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} sec")
    visualize_mt(torch.abs(mt_lh[random_number][Sensors.C3.value]), frequencies,
                 title='Left random hand mt magnitude C3')

    mt_rh = cwt1d(tensor(rh_data).to(device), scales, int_psi_scales=int_psi_scales, out_dtype='complex')
    mt_passive = cwt1d(tensor(passive_data).to(device), scales, int_psi_scales=int_psi_scales, out_dtype='complex')


    visualize_mt(torch.angle(mt_lh[random_number][Sensors.C3.value]), frequencies,
                 title='Left random hand mt phase C3')
    visualize_mt((mt_lh[random_number][Sensors.C3.value]), frequencies,
                 title='Left random hand mt representation C3')

    mt_lh_magn_mean = torch.mean(mt_lh, dim=0)
    mt_rh_magn_mean = torch.mean(mt_rh, dim=0)
    mt_passive_magn_mean = torch.mean(mt_passive, dim=0)

    visualize_sample(lh_data[random_number][Sensors.C3.value], title='Random single signal')
    lh_mean = np.mean(lh_data, axis=0)
    rh_mean = np.mean(rh_data, axis=0)
    visualize_sample(lh_mean[Sensors.C3.value], title='Mean C3 lh signal')
    visualize_sample(rh_mean[Sensors.C3.value], title='Mean C3 rh signal')

    visualize_mt(mt_lh_magn_mean[Sensors.C3.value], frequencies, title='Left hand mt C3')
    visualize_mt(mt_rh_magn_mean[Sensors.C3.value], frequencies, title='Right hand mt C3')
    visualize_mt(mt_passive_magn_mean[Sensors.C3.value], frequencies, title='Passive mt C3')

    visualize_mt(mt_lh_magn_mean[Sensors.Cz.value], frequencies, title='Left hand mt Cz')
    visualize_mt(mt_rh_magn_mean[Sensors.Cz.value], frequencies, title='Right hand mt Cz')
    visualize_mt(mt_passive_magn_mean[Sensors.Cz.value], frequencies, title='Passive mt Cz')

    visualize_mt(mt_lh_magn_mean[Sensors.C4.value], frequencies, title='Left hand mt C4')
    visualize_mt(mt_rh_magn_mean[Sensors.C4.value], frequencies, title='Right hand mt C4')
    visualize_mt(mt_passive_magn_mean[Sensors.C4.value], frequencies, title='Passive mt C4')

    # TODO: Проверить как выглядят преобразования на разных значениях b, обучить модель и померить качество
    print('sula')
