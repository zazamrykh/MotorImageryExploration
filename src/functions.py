import colorsys
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy
import torch
from scipy.signal import stft
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from src.wavelets.dwt1d import cwt1d

# matplotlib.use('TkAgg')

from src.params import number_of_channels, channels, imagery_actions, STFT_length, \
    STFT_overlap, numpy_dtype, wavelet, wt_frequencies, device, path_to_weights, spatial_channels_order


def visualize_sample(sample, marker=None, channels_to_show=None, title=None):
    plt.figure(figsize=(12, 7))
    if title is not None:
        plt.title(title)
    plt.xlabel('Time (sample number)')
    plt.ylabel('Voltage')
    plt.legend(loc='upper right')
    if len(sample.shape) == 1:
        plt.plot(sample)
    else:
        if channels_to_show is None:
            for i in range(number_of_channels):
                plt.plot(sample[i, :], label=f'Channel {channels[i]}')
        else:
            for i in range(channels_to_show):
                plt.plot(sample[i, :], label=f'Channel {channels[i]}')
    if marker is not None:
        plt.title(str(imagery_actions[marker]))

    plt.show()


def seed_everything(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def MorletTransform(signal, freq=wt_frequencies, wavelet=wavelet, sampling_rate=200, number_of_parts=1,
                    method='fft'):
    scales = pywt.frequency2scale(wavelet, freq / sampling_rate)
    if type(signal) == torch.Tensor:
        signal = signal.numpy()
    if len(signal.shape) == 4:
        chunks = []
        for i in range(0, signal.shape[0], signal.shape[0] / number_of_parts):
            chunks.append(signal[i:i + signal, :, :])
        cwt_result = [pywt.cwt(chunk, scales, wavelet, method=method)[0] for chunk in chunks]
        cwt_result = np.concatenate(cwt_result)
    else:
        cwt_result, frequencies = pywt.cwt(signal, scales, wavelet, method=method)
        cwt_result = cwt_result.astype(numpy_dtype)
    if len(cwt_result.shape) == 4:
        cwt_result = cwt_result.transpose(1, 2, 0, 3)
    if len(cwt_result.shape) == 3:
        cwt_result = cwt_result.transpose(1, 0, 2)
    return cwt_result


def visualize_mt(mt_result, frequencies, title=None):
    if mt_result.dtype == torch.complex64:
        amp = torch.abs(mt_result)
        phase = torch.angle(mt_result)
        H = phase / (2 * torch.pi)
        L = 1 - 2 ** (-amp)
        S = torch.ones_like(amp)

        repr = torch.stack([H, L, S])
        repr_np = repr.numpy()
        repr_np = np.moveaxis(repr_np, 0, -1)

        repr_np_rgb = np.zeros_like(repr_np)
        for i in range(repr_np.shape[0]):
            for j in range(repr_np.shape[1]):
                repr_np_rgb[i, j, :] = np.array(
                    colorsys.hls_to_rgb(repr_np[i, j, 0], repr_np[i, j, 1], repr_np[i, j, 2]))

        plt.imshow(repr_np_rgb, aspect='auto', extent=[0, 200, frequencies[-1], frequencies[0]], cmap='jet',
                   interpolation='bilinear')
        if title is None:
            plt.title(wavelet + 'wavelet transform')
        else:
            plt.title(title)
        plt.show()
        return
    plt.imshow(mt_result, aspect='auto', extent=[0, 200, frequencies[-1], frequencies[0]], cmap='jet',
               interpolation='bilinear')
    plt.colorbar(label='Magnitude')
    if title is None:
        plt.title(wavelet + 'wavelet transform')
    else:
        plt.title(title)
    plt.xlabel('Time axis')
    plt.ylabel('Frequency axis')
    plt.show()


def STFT(signal, sampling_rate=200, segm_length=STFT_length,
         noverlap=STFT_overlap):  # signal (22, 200) -> (22, N_freq, N_segm)
    # nperseg - lenght of each segm
    # noverlap - Number of points to overlap between segments. If None, noverlap = nperseg // 2.
    # Defaults to a Hann window so we change to hamming
    if type(signal) == torch.tensor:
        signal = signal.numpy()
    frequencies, times, magnitude = stft(signal, fs=sampling_rate, nperseg=segm_length,
                                         window='hamming', noverlap=noverlap)
    return frequencies, times, magnitude


def create_array_of_sections(indices, event_dur, eeg_data):
    array_of_motions = []
    for index in indices:
        if index + event_dur <= len(eeg_data):
            array_of_motions.append(eeg_data[index:index + event_dur])
    return np.array(array_of_motions)


def cut_all_imaginary_motion(path, from_mat=True, event_timestamps=200):
    if from_mat:
        mat = scipy.io.loadmat(path)
        markers = mat['o'][0][0][4]
        markers = np.reshape(markers, (-1))
        eeg_data = mat['o'][0][0][5]
    else:
        data_dict = None
        if path.endswith('.pkl'):
            with open(path, 'rb') as pickle_file:
                data_dict = pickle.load(pickle_file)
        markers = data_dict['marker']
        eeg_data = [data_dict[sensor] for sensor in channels]
        eeg_data = np.array(eeg_data)
        eeg_data = eeg_data.T

    differences = np.diff(markers)
    indices_lh = np.where(differences == 1)[0] + 1
    indices_rh = np.where(differences == 2)[0] + 1
    indices_passive = np.where(differences == 3)[0] + 1
    indices_ll = np.where(differences == 4)[0] + 1
    indices_t = np.where(differences == 5)[0] + 1
    indices_rl = np.where(differences == 6)[0] + 1

    lh_im = create_array_of_sections(indices_lh, event_timestamps, eeg_data)
    rh_im = create_array_of_sections(indices_rh, event_timestamps, eeg_data)
    passive_im = create_array_of_sections(indices_passive, event_timestamps, eeg_data)
    ll_im = create_array_of_sections(indices_ll, event_timestamps, eeg_data)
    t_im = create_array_of_sections(indices_t, event_timestamps, eeg_data)
    rl_im = create_array_of_sections(indices_rl, event_timestamps, eeg_data)

    arrays_list = [lh_im, rh_im, passive_im, ll_im, t_im, rl_im]
    markers = np.concatenate([np.full(len(arr), i + 1) for i, arr in enumerate(arrays_list)])
    data = np.concatenate(arrays_list, axis=0)

    return data, markers


def get_subject_imageries(subject, path):
    entries = os.listdir(path)

    all_data = []
    all_markers = []
    for entry in entries:
        full_path = os.path.join(path, entry)
        if 'Subject' + subject in full_path:
            if os.path.isfile(full_path):
                if '.mat' in full_path:
                    data, markers = cut_all_imaginary_motion(full_path)
                    all_data.append(data)
                    all_markers.append(markers)
                    break

    data = np.concatenate(all_data, axis=0)
    data = np.transpose(data, (0, 2, 1))
    markers = np.concatenate(all_markers, axis=0) - 1
    return data, markers


def get_all_imageries(path):
    entries = os.listdir(path)

    all_data = []
    all_markers = []
    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path):
            if '.mat' in full_path:
                data, markers = cut_all_imaginary_motion(full_path)
                all_data.append(data)
                all_markers.append(markers)
                break

    data = np.concatenate(all_data, axis=0)
    data = np.transpose(data, (0, 2, 1))
    markers = np.concatenate(all_markers, axis=0) - 1
    return data, markers


def normalize_data(data):
    means = np.mean(data, axis=(0, 2), keepdims=True)
    stds = np.std(data, axis=(0, 2), keepdims=True)
    return (data - means) / stds


def generate_mt_freq(points_num, bottom=0.1, top=40, power=1):
    output = []
    n = power
    c = bottom
    a = (top - c) / (top ** n)
    for x in np.linspace(0, top - bottom + 1, points_num):
        output.append(a * x ** n + c)
    return np.array(output)


def one_hot_encode(arr):
    unique_values = np.unique(arr)
    num_classes = len(unique_values)

    one_hot_tensor = torch.zeros((arr.shape[0], num_classes), dtype=torch.float32)

    for i, value in enumerate(arr):
        class_index = np.where(unique_values == value)[0][0]
        one_hot_tensor[i, class_index] = 1.0

    return one_hot_tensor


def create_dataloader(numpy_samples, numpy_target, batch_size=64):
    # Do one hot encoding
    if len(numpy_target.shape) == 1:
        torch_target = one_hot_encode(numpy_target).float()
    else:
        torch_target = torch.from_numpy(numpy_target).float()
    torch_samples = torch.from_numpy(numpy_samples).float()
    dataset = TensorDataset(torch_samples, torch_target)
    dataloader = DataLoader(dataset, batch_size)
    return dataloader


def create_dataset(numpy_samples, numpy_target):
    # Do one hot encoding
    if len(numpy_target.shape) == 1:
        torch_target = one_hot_encode(numpy_target).float()
    else:
        torch_target = torch.from_numpy(numpy_target).float()
    torch_samples = torch.from_numpy(numpy_samples).float()
    dataset = TensorDataset(torch_samples, torch_target)
    return dataset


def get_num_of_model_param(model):
    num_param = 0
    for parameter in model.parameters():
        num_param += parameter.numel()
    return num_param


def visualize_history(history, model_name):
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(path_to_weights + model_name + '.png')
    plt.show()


def evaluate_net(model, loader, loss_fn):
    model.eval()
    sum_loss = 0
    correct_predictions = 0
    total_examples = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model.forward(inputs)
        loss = loss_fn(outputs, labels)
        sum_loss += loss.item()

        predicted_classes = torch.argmax(outputs, dim=1)
        real_classes = torch.argmax(labels, dim=1)
        correct_predictions += torch.sum(real_classes == predicted_classes).item()
        total_examples += labels.shape[0]
    sum_loss = sum_loss / total_examples
    accuracy = correct_predictions / total_examples
    return sum_loss, accuracy


def change_channels_order(signal, new_order=spatial_channels_order):
    indices = [channels.index(channel) for channel in new_order]
    return signal[:, indices, :]


def seed_everything(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
