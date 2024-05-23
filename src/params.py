from enum import Enum

import numpy as np
import torch

sampling_rate = 200  # 200
hat_sampling_rate = 125
event_duration = 1.0
event_timestamps = sampling_rate * int(event_duration)
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8',
            'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'X5']
# channels = ['O1', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4', 'O2']

spatial_channels_order = ['Fp1', 'Fp2', 'F8', 'F4', 'Fz', 'F3', 'F7', 'A1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'A2',
                          'T6', 'P4', 'Pz', 'P3', 'T5', 'O1', 'O2', 'X5']
number_of_channels = 22
imagery_actions = ['left hand', 'right hand', 'passive', 'left leg', 'tongue', 'right leg']

path_to_dataset = '../Large EEG dataset/'
path_to_serialized = '../Large EEG dataset/serialized-datasets/'
path_to_weights = '../saved-weights/'

numpy_dtype = np.float32
torch_dtype = torch.float32
tensor_type = torch.FloatTensor
device = torch.device('cuda')
random_seed = 1337

wavelet = 'mexh'  # 'morl' #'cmor2.0-0.8' # 'fpsp2-3.0-1.0'
frequencies_num = 100#  60
batch_size = 128
epochs = 10
learning_rate = 0.0001
weight_decay = 0.01
dropout_rate = 0.1
out_dtype = 'real'
bottom = 1
top = frequencies_num
power = 2
step_size = 5
gamma = 0.2

STFT_freq_num = 50
STFT_length = STFT_freq_num * 2 - 1  # 99
STFT_time_num = 40
STFT_overlap = int(abs((sampling_rate - STFT_time_num * (STFT_length - 1)) / (STFT_time_num - 1))) - 1  # 94


class Imagery(Enum):
    LEFT_HAND = 0
    RIGHT_HAND = 1
    PASSIVE = 2
    LEFT_LEG = 3
    TONGUE = 4
    RIGHT_LEG = 5


class Sensors(Enum):
    Fp1 = 0
    Fp2 = 1
    F3 = 2
    F4 = 3
    C3 = 4
    C4 = 5
    P3 = 6
    P4 = 7
    O1 = 8
    O2 = 9
    A1 = 10
    A2 = 11
    F7 = 12
    F8 = 13
    T3 = 14
    T4 = 15
    T5 = 16
    T6 = 17
    Fz = 18
    Cz = 19
    Pz = 20
    X5 = 21


all_sensors = [Sensors(i) for i in range(number_of_channels)]
