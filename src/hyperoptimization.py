import json

import numpy as np
import optuna
import pywt
import torch
from torch import nn
from torch.optim import lr_scheduler

from src.functions import create_dataloader, generate_mt_freq, evaluate_net
from src.network import WTCNN3D, WTConvNet
from src.params import path_to_serialized, sampling_rate, device, channels, path_to_weights
from src.train import train
from src.wavelets.dwt1d import generate_int_psi_scales


def optimize():
    study = optuna.create_study(study_name="my_first_study", direction="maximize")

    data_train = np.load(path_to_serialized + 'data_train.npy')
    data_val = np.load(path_to_serialized + 'data_valid.npy')
    markers_train = np.load(path_to_serialized + 'markers_train.npy')
    markers_val = np.load(path_to_serialized + 'markers_valid.npy')

    # Объединение данных и меток
    data = np.concatenate((data_train, data_val), axis=0)
    markers = np.concatenate((markers_train, markers_val), axis=0)

    # Освобождение памяти
    del data_train, data_val, markers_train, markers_val

    # Разделение данных на две равные части
    half_size = data.shape[0] // 2
    data_first_half = data[:half_size]
    markers_first_half = markers[:half_size]

    # Освобождение памяти
    del data, markers

    # Разделение первой половины на тренировочную и тестовую части (80:20)
    train_size = int(half_size * 0.8)

    data_train = data_first_half[:train_size]
    markers_train = markers_first_half[:train_size]

    data_val = data_first_half[train_size:]
    markers_val = markers_first_half[train_size:]

    # Освобождение памяти
    del data_first_half, markers_first_half

    def objective(trial):
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        wavelet = trial.suggest_categorical('wavelet', ['mexh', 'morl'])
        frequencies_num = trial.suggest_categorical('frequencies_num', [80, 100])
        power = trial.suggest_categorical('power', [1, 2, 3])
        epochs = 5
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-2, 3e-1, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.3)
        step_size = trial.suggest_int("step_size", 4, 8)
        gamma = trial.suggest_float("gamma", 0.05, 0.7)

        train_loader = create_dataloader(data_train, markers_train, batch_size)
        val_loader = create_dataloader(data_val, markers_val, batch_size)

        frequencies = generate_mt_freq(frequencies_num, bottom=1, top=frequencies_num, power=power)
        scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
        int_psi_scales = generate_int_psi_scales(scales, wavelet, device)
        model_name = 'WTCNN3D' + '_' + wavelet + '_bs' + str(batch_size) + '_ep' + str(
            epochs) + '_lr' + str(
            learning_rate) + '_wd' + str(weight_decay) + '_dr' + str(dropout_rate)
        model = WTConvNet(model_name, scales, int_psi_scales, frequencies_num=frequencies_num, input_timestamps=sampling_rate, dropout=dropout_rate,
                          channels_number=len(channels)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        loss = nn.BCELoss(reduction='sum')

        history = train(model, train_loader, val_loader, loss, epochs, optimizer, scheduler, early_stop=True, output=False)
        test_loss, test_accuracy = evaluate_net(model, val_loader, loss)

        if (trial.number + 1) % 5 == 0:
            with open(path_to_weights + 'best_params_hyperoptimization.json', 'w') as f:
                json.dump(study.best_params, f)

        return test_accuracy

    study.optimize(objective, n_trials=500)
    print(study.best_params)
    best_params = study.best_params
    with open(path_to_weights + 'best_params_hyperoptimization.json', 'w') as f:
        json.dump(best_params, f)

    print("Best parameters saved to 'best_params.json'")



if __name__ == '__main__':
    optimize()