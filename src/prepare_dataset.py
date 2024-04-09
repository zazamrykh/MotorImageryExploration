import numpy as np

from src.functions import get_all_imageries
from src.params import path_to_dataset, path_to_serialized
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data, markers = get_all_imageries(path_to_dataset)
    random_seed = 42
    np.random.seed(random_seed)

    random_indices = np.arange(len(data))
    np.random.shuffle(random_indices)

    shuffled_data = data[random_indices]
    shuffled_markers = markers[random_indices]

    small_data = data[0:1000]
    small_markers = markers[0:1000]

    train_size = 0.6
    test_size = 0.2
    valid_size = 0.2

    data_train, data_temp, markers_train, markers_temp = train_test_split(
        data, markers, train_size=train_size, stratify=markers, random_state=42)

    data_test, data_valid, markers_test, markers_valid = train_test_split(
        data_temp, markers_temp, test_size=valid_size/(test_size + valid_size), stratify=markers_temp, random_state=42)

    means = np.mean(data_train, axis=(0, 2), keepdims=True)
    stds = np.std(data_train, axis=(0, 2), keepdims=True)

    data_train = (data_train - means) / stds
    data_test = (data_test - means) / stds
    data_valid = (data_valid - means) / stds

    np.save(path_to_serialized + 'data_train_small.npy', data_train)
    np.save(path_to_serialized + 'data_test_small.npy', data_test)
    np.save(path_to_serialized + 'data_valid_small.npy', data_valid)

    np.save(path_to_serialized + 'markers_train_small.npy', markers_train)
    np.save(path_to_serialized + 'markers_test_small.npy', markers_test)
    np.save(path_to_serialized + 'markers_valid_small.npy', markers_valid)