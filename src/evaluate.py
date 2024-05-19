import matplotlib
import numpy as np
import pywt
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.functions import generate_mt_freq, create_dataloader
from src.network import WTConvNet
from src.params import wavelet, sampling_rate, device, path_to_serialized, imagery_actions
from src.wavelets.dwt1d import generate_int_psi_scales
from PIL import ImageTk

if __name__ == '__main__':
    w8s_path = '../saved-weights/0.713_MWTCNN_mexh_bs_64_ep_25_lr_0.0001_wd_0.1_dr_0.3.pth.pth'
    points_num = 80
    frequencies = generate_mt_freq(points_num, bottom=1, top=80, power=2)
    scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    model_name = 'WTCNN_test'
    model = WTConvNet(model_name, scales, int_psi_scales).to(device)

    model.load_state_dict(torch.load(w8s_path))

    batch_size = 64
    data_test = np.load(path_to_serialized + 'data_test.npy')
    markers_test = np.load(path_to_serialized + 'markers_test.npy')
    test_loader = create_dataloader(data_test, markers_test, batch_size)
    loss_fn = nn.BCELoss(reduction='sum')

    model.eval()
    predicted_list = []
    real_list = []
    with torch.no_grad():
        sum_loss = 0
        correct_predictions = 0
        total_examples = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = loss_fn(outputs, labels)
            sum_loss += loss.item()

            predicted_classes = torch.argmax(outputs, dim=1)
            real_classes = torch.argmax(labels, dim=1)
            total_examples += labels.shape[0]
            correct_predictions += torch.sum(real_classes == predicted_classes).item()
            predicted_list.append(predicted_classes)
            real_list.append(real_classes)
        accuracy = correct_predictions / total_examples
        loss = sum_loss / total_examples

    predicted_list = torch.cat(predicted_list).tolist()
    real_list = torch.cat(real_list).tolist()
    conf_matrix = confusion_matrix(real_list, predicted_list)
    print(conf_matrix)
    plt.figure(figsize=(12, 7))
    plt.imshow(conf_matrix, cmap='viridis', interpolation='nearest')

    plt.title('Confusion Matrix')
    plt.colorbar(label='Number of Occurrences')
    plt.xticks(ticks=np.arange(len(imagery_actions)), labels=imagery_actions, rotation=45)
    plt.yticks(ticks=np.arange(len(imagery_actions)), labels=imagery_actions)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Добавляем текст в каждую ячейку
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()
