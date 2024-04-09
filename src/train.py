import copy
import gc

import numpy as np
import pywt
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import lr_scheduler

from src.functions import create_dataloader, get_num_of_model_param, visualize_history, generate_mt_freq, evaluate_net
from src.network import MWTConvNet, WaveletTransform
from src.params import path_to_serialized, device, path_to_saved_weights, wavelet, sampling_rate, out_dtype
from wavelets.dwt1d import generate_int_psi_scales


def compute_l1_loss(w):
    return torch.abs(w).sum()


def compute_l2_loss(w):
    return torch.square(w).sum()


def train(model, data_tr, data_val, loss_fn, epochs, optimizer, scheduler=None, freeze_moment=None, output=True):
    if output: print('Start training')
    history = {'train': [], 'val': [], 'accuracy': []}
    best_accuracy = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        # For transfer learning
        if epoch == freeze_moment:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

        avg_loss = 0
        model.train()
        for X_batch, Y_batch in data_tr:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model.forward(X_batch)
            loss = loss_fn(Y_pred, Y_batch)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_loss /= len(data_tr)
        history['train'].append(avg_loss)

        model.eval()
        avg_val_loss = 0
        accuracy = 0
        for inputs, labels in data_val:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = loss_fn(outputs, labels)
            avg_val_loss += loss.item()

            predicted_classes = torch.argmax(outputs, dim=1)
            real_classes = torch.argmax(labels, dim=1)
            correct_predictions = torch.sum(real_classes == predicted_classes).item()
            total_examples = labels.shape[0]
            accuracy += correct_predictions / total_examples
        avg_val_loss /= len(data_val)
        accuracy /= len(data_val)
        history['val'].append(avg_val_loss)
        history['accuracy'].append(accuracy)
        if output:
            print('%d / %d - val loss: %f, train loss: %f, accuracy: %f' % (epoch + 1, epochs,
                                                                            avg_val_loss, avg_loss, accuracy))

        if epoch == 0 or accuracy > best_accuracy:
            best_accuracy = accuracy
            model.best_accuracy = best_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path_to_saved_weights + model.name + '.pth')

        if scheduler is not None:
            scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

    model.load_state_dict(best_model_wts)
    return history


def train_MWTCNN():
    data_train = np.load(path_to_serialized + 'data_train.npy')
    data_test = np.load(path_to_serialized + 'data_test.npy')
    data_val = np.load(path_to_serialized + 'data_valid.npy')
    markers_train = np.load(path_to_serialized + 'markers_train.npy')
    markers_test = np.load(path_to_serialized + 'markers_test.npy')
    markers_val = np.load(path_to_serialized + 'markers_valid.npy')

    batch_size = 64
    epochs = 25
    class_num = 6
    learinig_rate = 1e-4
    weight_decay = 0.1
    dropout = 0.3
    train_loader = create_dataloader(data_train, markers_train, batch_size)
    val_loader = create_dataloader(data_val, markers_val, batch_size)
    test_loader = create_dataloader(data_test, markers_test, batch_size)

    points_num = 80
    frequencies = generate_mt_freq(points_num, bottom=1, top=80, power=2)
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    model_name = 'MWTCNN' + '|' + wavelet + '|bs:' + str(batch_size) + '|ep:' + str(epochs) + '|lr:' + str(
        learinig_rate) + '|wd:' + str(weight_decay) + '|dr:' + str(dropout)
    model = MWTConvNet(model_name, scales, int_psi_scales, dropout=dropout).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learinig_rate, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learinig_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss = nn.BCELoss()
    number_of_param = get_num_of_model_param(model)
    print('Number of model params:', number_of_param)

    history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler)

    test_loss, test_accuracy = evaluate_net(model, test_loader, loss)
    print('Result test loss:', test_loss, '\nResult test accuracy:', test_accuracy)

    model_name = str(round(test_accuracy, 3)) + '|' + model.name + '.pth'
    visualize_history(history, model_name)
    torch.save(model.state_dict(), path_to_saved_weights + model_name + '.pth')


def train_resnet():
    data_train = np.load(path_to_serialized + 'data_train.npy')
    data_test = np.load(path_to_serialized + 'data_test.npy')
    data_val = np.load(path_to_serialized + 'data_valid.npy')
    markers_train = np.load(path_to_serialized + 'markers_train.npy')
    markers_test = np.load(path_to_serialized + 'markers_test.npy')
    markers_val = np.load(path_to_serialized + 'markers_valid.npy')

    batch_size = 64
    epochs = 25
    class_num = 6
    learinig_rate = 4e-4
    weight_decay = 0.1
    dropout = 0.0
    train_loader = create_dataloader(data_train, markers_train, batch_size)
    val_loader = create_dataloader(data_val, markers_val, batch_size)
    test_loader = create_dataloader(data_test, markers_test, batch_size)

    points_num = 80
    frequencies = generate_mt_freq(points_num, bottom=1, top=80, power=2)
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

    model_name = str(round(test_accuracy, 3)) + '|' + 'resnet18' + '|' + wavelet + '|bs:' + str(batch_size) + '|ep:' + str(epochs) + '|lr:' + str(
        learinig_rate) + '|wd:' + str(weight_decay) + '|dr:' + str(dropout) + '.pth'
    visualize_history(history, model_name)
    torch.save(model.state_dict(), path_to_saved_weights + model_name + '.pth')


if __name__ == '__main__':
    neural_network = 'MWTCNN'
    if neural_network == 'MWTCNN':
        train_MWTCNN()
    if neural_network == 'resnet18':
        train_resnet()
