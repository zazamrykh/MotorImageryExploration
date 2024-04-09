import gc
import os

import numpy as np
import pywt
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import lr_scheduler

from src.functions import cut_all_imaginary_motion, create_dataloader, generate_mt_freq, get_num_of_model_param, \
    evaluate_net
from src.network import MWTConvNet
from src.params import sampling_rate, device, path_to_saved_weights, wavelet
from src.train import train
from wavelets.dwt1d import generate_int_psi_scales

if __name__ == '__main__':
    subjects_list = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    test_losses = []
    test_accuracies = []

    train_size = 0.75
    test_size = 0.13
    valid_size = 0.12

    for subject_letter in subjects_list:
        # sry for copepaste ;-(
        directory_path = './Large EEG dataset/'
        entries = os.listdir(directory_path)
        event_duration = sampling_rate
        all_data = []
        all_markers = []
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if 'Subject' + subject_letter in full_path:
                if os.path.isfile(full_path):
                    if '.mat' in full_path:
                        data, markers = cut_all_imaginary_motion(full_path)
                        all_data.append(data)
                        all_markers.append(markers)

        data = np.concatenate(all_data, axis=0)
        data = np.transpose(data, (0, 2, 1))
        markers = np.concatenate(all_markers, axis=0)
        _ = gc.collect()

        data_train, data_temp, markers_train, markers_temp = train_test_split(
            data, markers, train_size=train_size, stratify=markers, random_state=42)

        data_test, data_valid, markers_test, markers_valid = train_test_split(
            data_temp, markers_temp, test_size=valid_size / (test_size + valid_size),
            stratify=markers_temp, random_state=42)

        means = np.mean(data_train, axis=(0, 2), keepdims=True)
        stds = np.std(data_train, axis=(0, 2), keepdims=True)
        data_train = (data_train - means) / stds
        data_test = (data_test - means) / stds
        data_valid = (data_valid - means) / stds

        batch_size = 64
        epochs = 18
        class_num = 6
        learinig_rate = 1e-4
        weight_decay = 0.1
        dropout = 0.3

        train_loader = create_dataloader(data_train, markers_train, batch_size)
        val_loader = create_dataloader(data_valid, markers_valid, batch_size)
        test_loader = create_dataloader(data_test, markers_test, batch_size)

        points_num = 80
        frequencies = generate_mt_freq(points_num, bottom=1, top=80, power=2)
        scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
        int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

        model_name = 'MWTCNN' + '|' + wavelet + '|bs:' + str(batch_size) + '|ep:' + str(epochs) + '|lr:' + str(
            learinig_rate) + '|wd:' + str(weight_decay) + '|dr:' + str(dropout)
        model = MWTConvNet(model_name, scales, int_psi_scales, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learinig_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        loss = nn.BCELoss()

        history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler, output=False)
        test_loss, test_accuracy = evaluate_net(model, test_loader, loss)
        print('Subject ' + subject_letter + ' test loss:', test_loss, 'test accuracy:', test_accuracy)

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    print('Test losses:')
    print(["{:.3f}".format(loss) for loss in test_losses])
    print('Test accuracies:')
    print(["{:.3f}".format(accuracy) for accuracy in test_accuracies])
    print('Mean test loss:', np.mean(test_losses))
    print('Mean test accuracy:', np.mean(test_accuracies))
    print('Median test accuracy:', np.median(test_accuracies))
    print('Minimum test accuracy:', np.min(test_accuracies))
    print('Maximum test accuracy:', np.max(test_accuracies))

    accuracy = np.array(test_accuracies)
    step = 0.1

    bins = np.arange(min(accuracy), max(accuracy) + step, step)

    plt.hist(accuracy, bins=bins, edgecolor='black')
    plt.title('Histogram of accuracies')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')

    plt.savefig(path_to_saved_weights + 'individual.png')
    plt.show()