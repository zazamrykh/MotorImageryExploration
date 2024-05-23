import copy
import gc
import os.path

import numpy as np
import optuna
import pywt
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.functions import create_dataloader, get_num_of_model_param, visualize_history, evaluate_net, \
    create_dataset, seed_everything, generate_mt_freq
from src.network import WTConvNet, WaveletTransform, WTCNN3D
from src.params import path_to_serialized, device, path_to_weights, wavelet, sampling_rate, out_dtype, \
    random_seed, batch_size, epochs, dropout_rate, weight_decay, learning_rate, frequencies_num, step_size, \
    gamma, channels
from wavelets.dwt1d import generate_int_psi_scales


def evaluate(model, dataloader, loss_fn):
    model.eval()
    sum_loss = 0
    correct_predictions = 0
    total_examples = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model.forward(inputs)
        loss = loss_fn(outputs, labels)
        sum_loss += loss.item()

        predicted_classes = torch.argmax(outputs, dim=1)
        real_classes = torch.argmax(labels, dim=1)
        total_examples += labels.shape[0]
        correct_predictions += torch.sum(real_classes == predicted_classes).item()
    accuracy = correct_predictions / total_examples
    loss = sum_loss / total_examples
    return loss, accuracy


def train(model, data_tr, data_val, loss_fn, epochs, optimizer, scheduler=None, freeze_moment=None, output=True,
          early_stop=True):
    if output: print('Start training')
    history = {'train': [], 'val': [], 'accuracy': [], 'train_accuracy': []}
    best_accuracy = 0
    best_loss = 1_000_000_000 #  just big value
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        # For transfer learning
        if epoch == freeze_moment:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

        model.train()
        correct_predictions = 0
        total_examples = 0
        sum_loss = 0
        for X_batch, Y_batch in data_tr:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model.forward(X_batch)
            loss = loss_fn(Y_pred, Y_batch)

            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            total_examples += Y_batch.shape[0]

            predicted_classes = torch.argmax(Y_pred, dim=1)
            real_classes = torch.argmax(Y_batch, dim=1)
            correct_predictions += torch.sum(real_classes == predicted_classes).item()
        train_accuracy = correct_predictions / total_examples
        train_loss = sum_loss / total_examples
        history['train'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        if data_val is not None:
            val_loss, val_accuracy = evaluate(model, data_val, loss_fn)
            history['val'].append(val_loss)
            history['accuracy'].append(val_accuracy)
            if output:
                print('%d / %d - val loss: %f, train loss: %f, accuracy: %f' % (epoch + 1, epochs,
                                                                                val_loss, train_loss, val_accuracy))
            if epoch == 0 or val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
                best_accuracy = val_accuracy
                best_loss = val_loss
                model.best_accuracy = best_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            if output:
                print('%d / %d - train loss: %f, train accuracy: %f' % (epoch + 1, epochs,
                                                                        train_loss, train_accuracy))

        if scheduler is not None:
            scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

    if early_stop:
        model.load_state_dict(best_model_wts)
    return history


def load_all_data(load_exist=True):
    if not load_exist or not (
            os.path.exists(path_to_serialized + 'data.npy') and os.path.exists(path_to_serialized + 'markers.npy')):
        data_train = np.load(path_to_serialized + 'data_train.npy')
        data_test = np.load(path_to_serialized + 'data_test.npy')
        data_val = np.load(path_to_serialized + 'data_valid.npy')
        markers_train = np.load(path_to_serialized + 'markers_train.npy')
        markers_test = np.load(path_to_serialized + 'markers_test.npy')
        markers_val = np.load(path_to_serialized + 'markers_valid.npy')

        all_data = np.concatenate((data_train, data_test, data_val), axis=0)
        all_markers = np.concatenate((markers_train, markers_test, markers_val), axis=0)

        np.save(path_to_serialized + 'data.npy', all_data)
        np.save(path_to_serialized + 'markers.npy', all_markers)
    else:
        all_data = np.load(path_to_serialized + 'data.npy')
        all_markers = np.load(path_to_serialized + 'markers.npy')
    return all_data, all_markers


def cross_validation(data, labels):
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    if data is None or labels is None:
        data, labels = load_all_data(False)
    dataset = create_dataset(data, labels)

    batch_size = 64
    epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.1
    dropout = 0.3
    points_num = 80
    bottom = 1
    top = 80
    power = 2
    step_size = 5
    gamma = 0.2
    val_split = 0.12

    frequencies = generate_mt_freq(frequencies_num)
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    accuracies = []
    losses = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")
        num_train_samples = len(train_idx)
        indices = list(range(num_train_samples))
        split = int(np.floor(val_split * num_train_samples))

        # Shuffle the indices if needed
        np.random.shuffle(indices)

        train_indices = indices[split:]
        val_indices = indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=val_sampler,
        )
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        model = WTConvNet('cv_wt_net', scales, int_psi_scales, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        loss = nn.BCELoss(reduction='sum')

        history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler)
        test_loss, test_accuracy = evaluate(model, test_loader, loss)
        accuracies.append(test_accuracy)
        losses.append(test_loss)
        print('%d / %d - test_loss: %f, test_accuracy: %f' % (fold + 1, k_folds, test_loss, test_accuracy))

    print('Accuracies:', accuracies)
    print('Mean:', np.mean(accuracies))
    print('Median:', np.median(accuracies))
    with open(path_to_weights + 'cv_metrics1.txt', 'w') as f:
        f.write('batch_size = {}\n'.format(batch_size))
        f.write('epochs = {}\n'.format(epochs))
        f.write('learning_rate = {}\n'.format(learning_rate))
        f.write('weight_decay = {}\n'.format(weight_decay))
        f.write('dropout = {}\n'.format(dropout))
        f.write('points_num = {}\n'.format(points_num))
        f.write('bottom = {}\n'.format(bottom))
        f.write('top = {}\n'.format(top))
        f.write('power = {}\n'.format(power))
        f.write('step_size = {}\n'.format(step_size))
        f.write('gamma = {}\n'.format(gamma))
        f.write('val_split = {}\n'.format(val_split))
        f.write('Accuracies:\n')
        for accuracy in accuracies:
            f.write('{}\n'.format(accuracy))
        f.write('Mean: {}\n'.format(np.mean(accuracies)))
        f.write('Median: {}\n'.format(np.median(accuracies)))
        f.write('Mean loss: {}\n'.format(np.mean(losses)))
    return np.mean(losses), np.mean(accuracies)


def train_WTCNN():
    join_train_test = False
    head_sampling_rate = 125
    do_decreasing_sr = False
    do_decreasing_ch = False

    # data_train = np.load(path_to_serialized + 'my-dataset/data_train.npy')
    # data_test = np.load(path_to_serialized + 'my-dataset/data_test.npy')
    # data_val = np.load(path_to_serialized + 'my-dataset/data_val.npy')
    # markers_train = np.load(path_to_serialized + 'my-dataset/markers_train.npy')
    # markers_test = np.load(path_to_serialized + 'my-dataset/markers_test.npy')
    # markers_val = np.load(path_to_serialized + 'my-dataset/markers_val.npy')

    data_train = np.load(path_to_serialized + 'data_train.npy')
    data_test = np.load(path_to_serialized + 'data_test.npy')
    data_val = np.load(path_to_serialized + 'data_valid.npy')
    markers_train = np.load(path_to_serialized + 'markers_train.npy')
    markers_test = np.load(path_to_serialized + 'markers_test.npy')
    markers_val = np.load(path_to_serialized + 'markers_valid.npy')

    if join_train_test:
        data_train = np.concatenate((data_train, data_test), axis=0)
        data_test = data_val
        markers_train = np.concatenate((markers_train, markers_test), axis=0)
        markers_test = markers_val

    if do_decreasing_ch:
        channels_values = [sensor.value for sensor in channels]
        data_train = data_train[:, channels_values, :]
        data_val = data_val[:, channels_values, :]
        data_test = data_test[:, channels_values, :]

    if do_decreasing_sr:
        step = sampling_rate / head_sampling_rate
        # Increase sampling rate
        data_train = data_train[:, :, [int(i * step) for i in range(0, head_sampling_rate)]]
        data_val = data_val[:, :, [int(i * step) for i in range(0, head_sampling_rate)]]
        data_test = data_test[:, :, [int(i * step) for i in range(0, head_sampling_rate)]]

    train_loader = create_dataloader(data_train, markers_train, batch_size)
    val_loader = create_dataloader(data_val, markers_val, batch_size)
    test_loader = create_dataloader(data_test, markers_test, batch_size)

    frequencies = generate_mt_freq(frequencies_num)
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    model_name = 'WTCNN' + '_' + wavelet + '_bs' + str(batch_size) + '_ep' + str(
        epochs) + '_lr' + str(
        learning_rate) + '_wd' + str(weight_decay) + '_dr' + str(dropout_rate)
    model = WTConvNet(model_name, scales, int_psi_scales, input_timestamps=sampling_rate, dropout=dropout_rate,
                      channels_number=len(channels)).to(device)
    # model.load_state_dict(torch.load(path_to_weights + "0.546_WTCNN_pretrained_125ts_60freq_mexh_bs64_ep10_lr0.0001_wd0.01_dr0.1.pth"))

    # optimizer = torch.optim.SGD(model.parameters(), lr=learinig_rate, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss = nn.BCELoss(reduction='sum')
    number_of_param = get_num_of_model_param(model)
    print('Number of model params:', number_of_param)

    history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler, early_stop=True)
    test_loss, test_accuracy = evaluate_net(model, test_loader, loss)
    print('Result test loss:', test_loss, '\nResult test accuracy:', test_accuracy)

    model_name = str(round(test_accuracy, 3)) + '_' + model.name
    torch.save(model.state_dict(), path_to_weights + model_name)
    visualize_history(history, model_name, normalized=True)


def train_WTCNN3D():
    data_train = np.load(path_to_serialized + 'data_train.npy')
    data_test = np.load(path_to_serialized + 'data_test.npy')
    data_val = np.load(path_to_serialized + 'data_valid.npy')
    markers_train = np.load(path_to_serialized + 'markers_train.npy')
    markers_test = np.load(path_to_serialized + 'markers_test.npy')
    markers_val = np.load(path_to_serialized + 'markers_valid.npy')
    train_loader = create_dataloader(data_train, markers_train, batch_size)
    val_loader = create_dataloader(data_val, markers_val, batch_size)
    test_loader = create_dataloader(data_test, markers_test, batch_size)

    frequencies = generate_mt_freq(frequencies_num)
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    model_name = 'WTCNN3D' + '_' + wavelet + '_bs' + str(batch_size) + '_ep' + str(
        epochs) + '_lr' + str(
        learning_rate) + '_wd' + str(weight_decay) + '_dr' + str(dropout_rate)
    model = WTCNN3D(scales, int_psi_scales, name=model_name, dropout=dropout_rate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss = nn.BCELoss(reduction='sum')
    number_of_param = get_num_of_model_param(model)
    print('Number of model params:', number_of_param)

    history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler, early_stop=True)
    test_loss, test_accuracy = evaluate_net(model, test_loader, loss)
    print('Result test loss:', test_loss, '\nResult test accuracy:', test_accuracy)

    model_name = str(round(test_accuracy, 3)) + '_' + model.name
    torch.save(model.state_dict(), path_to_weights + model_name + '.pth')
    visualize_history(history, model_name, normalized=True)


def train_resnet():
    data_train = np.load(path_to_serialized + 'data_train.npy')
    data_test = np.load(path_to_serialized + 'data_test.npy')
    data_val = np.load(path_to_serialized + 'data_valid.npy')
    markers_train = np.load(path_to_serialized + 'markers_train.npy')
    markers_test = np.load(path_to_serialized + 'markers_test.npy')
    markers_val = np.load(path_to_serialized + 'markers_valid.npy')

    batch_size = 64
    epochs = 25
    learinig_rate = 4e-4
    weight_decay = 0.1
    dropout = 0.0
    train_loader = create_dataloader(data_train, markers_train, batch_size)
    val_loader = create_dataloader(data_val, markers_val, batch_size)
    test_loader = create_dataloader(data_test, markers_test, batch_size)

    frequencies = generate_mt_freq(frequencies_num)
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.conv1 = nn.Sequential(
        WaveletTransform(scales, int_psi_scales, out_dtype=out_dtype),
        nn.Conv2d(22, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learinig_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss = nn.BCELoss()
    number_of_param = get_num_of_model_param(model)
    print('Number of model params:', number_of_param)

    history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler)

    test_loss, test_accuracy = evaluate_net(model, test_loader, loss)
    print('Result test loss:', test_loss, '\nResult test accuracy:', test_accuracy)

    model_name = str(round(test_accuracy, 3)) + '_' + 'resnet18_' + wavelet + 'bs_' + str(batch_size) + 'ep_' + str(
        epochs) + 'lr_' + str(
        learinig_rate) + 'wd_' + str(weight_decay) + 'dr_' + str(dropout) + '.pth'
    torch.save(model.state_dict(), path_to_weights + model_name + '.pth')
    visualize_history(history, model_name)



if __name__ == '__main__':
    neural_network = 'WTCNN'
    do_cross_validation = False
    seed_everything(random_seed)
    if neural_network == 'WTCNN':
        if do_cross_validation:
            cross_validation(None, None)
        else:
            train_WTCNN()
    elif neural_network == 'resnet18':
        train_resnet()
    elif neural_network == 'WT3DCNN':
        train_WTCNN3D()
