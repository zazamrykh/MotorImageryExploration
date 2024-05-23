import gc
import os

import numpy as np
import pywt
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import lr_scheduler

from src.functions import cut_all_imaginary_motion, create_dataloader, generate_mt_freq, evaluate_net
from src.network import WTConvNet
from src.params import sampling_rate, device, path_to_weights, wavelet, path_to_dataset, learning_rate, batch_size, \
    epochs, dropout_rate, weight_decay, frequencies_num, bottom, top, power, step_size, gamma
from src.train import train, cross_validation
from wavelets.dwt1d import generate_int_psi_scales

if __name__ == '__main__':
    subjects_list = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    test_losses = []
    test_accuracies = []

    train_size = 0.75
    test_size = 0.13
    valid_size = 0.12
    do_cross_validation = True

    for subject_letter in subjects_list:
        # sry for copepaste ;-(
        directory_path = path_to_dataset
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

        if do_cross_validation:
            loss, accuracy = cross_validation(data, markers)
            test_losses.append(loss)
            test_accuracies.append(accuracy)
        else:
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

            train_loader = create_dataloader(data_train, markers_train, batch_size)
            val_loader = create_dataloader(data_valid, markers_valid, batch_size)
            test_loader = create_dataloader(data_test, markers_test, batch_size)

            frequencies = generate_mt_freq(frequencies_num, bottom=bottom, top=top, power=power)
            scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
            int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

            model_name = 'WTCNN_IT' + '_' + wavelet + '_bs' + str(batch_size) + '_ep' + str(epochs) + '_lr' + str(
                learning_rate) + '_wd' + str(weight_decay) + '_dr' + str(dropout_rate)
            model = WTConvNet(model_name, scales, int_psi_scales, dropout=dropout_rate).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            loss = nn.BCELoss()

            history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler, output=True)
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

    with open(path_to_weights + 'cv_metrics2.txt', 'w') as f:
        f.write('batch_size = {}\n'.format(batch_size))
        f.write('epochs = {}\n'.format(epochs))
        f.write('learning_rate = {}\n'.format(learning_rate))
        f.write('weight_decay = {}\n'.format(weight_decay))
        f.write('dropout = {}\n'.format(dropout_rate))
        f.write('points_num = {}\n'.format(frequencies_num))
        f.write('bottom = {}\n'.format(bottom))
        f.write('top = {}\n'.format(top))
        f.write('power = {}\n'.format(power))
        f.write('step_size = {}\n'.format(step_size))
        f.write('gamma = {}\n'.format(gamma))
        f.write('Accuracies:\n')
        for accuracy in test_accuracies:
            f.write('{}\n'.format(accuracy))
        f.write('Mean: {}\n'.format(np.mean(test_accuracies)))
        f.write('Median: {}\n'.format(np.median(test_accuracies)))
        f.write('Mean loss: {}\n'.format(np.mean(test_losses)))

    accuracy = np.array(test_accuracies)
    step = 0.1

    bins = np.arange(min(accuracy), max(accuracy) + step, step)

    plt.hist(accuracy, bins=bins, edgecolor='black')
    plt.title('Histogram of accuracies')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')

    plt.savefig(path_to_weights + 'individual.png')
    plt.show()
