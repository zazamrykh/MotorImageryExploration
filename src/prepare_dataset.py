import numpy as np

from src.functions import get_all_imageries, seed_everything, cut_all_imaginary_motion, visualize_sample
from src.params import path_to_dataset, path_to_serialized, random_seed
from sklearn.model_selection import train_test_split
import pickle


def main():
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
        data_temp, markers_temp, test_size=valid_size / (test_size + valid_size), stratify=markers_temp,
        random_state=42)

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


def prepare(path_to_files):
    seed_everything(random_seed)

    data_list = []
    markers_list = []
    for path_to_file in path_to_files:
        data_one_file, markers_one_file = cut_all_imaginary_motion(path_to_file, from_mat=False, event_timestamps=125)
        data_list.append(data_one_file)
        markers_list.append(markers_one_file)

    data = np.concatenate(data_list, axis=0)
    markers = np.concatenate(markers_list, axis=0)
    data = np.transpose(data, (0, 2, 1))
    train_size = 0.7
    test_size = 0.15
    val_size = 0.15

    data_train, data_temp, markers_train, markers_temp = train_test_split(
        data, markers, train_size=train_size, stratify=markers, random_state=random_seed)

    data_test, data_valid, markers_test, markers_valid = train_test_split(
        data_temp, markers_temp, test_size=val_size / (test_size + val_size), stratify=markers_temp,
        random_state=random_seed)

    means = np.mean(data_train, axis=(0, 2), keepdims=True)
    stds = np.std(data_train, axis=(0, 2), keepdims=True)

    data_train = (data_train - means) / stds
    data_test = (data_test - means) / stds
    data_valid = (data_valid - means) / stds

    np.save(path_to_serialized + 'my-dataset/data_train.npy', data_train)
    np.save(path_to_serialized + 'my-dataset/data_test.npy', data_test)
    np.save(path_to_serialized + 'my-dataset/data_val.npy', data_valid)

    np.save(path_to_serialized + 'my-dataset/markers_train.npy', markers_train)
    np.save(path_to_serialized + 'my-dataset/markers_test.npy', markers_test)
    np.save(path_to_serialized + 'my-dataset/markers_val.npy', markers_valid)
    visualize_sample(data_train[17][3])


if __name__ == '__main__':
    # main()
    records = ['./live/records/hundred_1.pkl', './live/records/hundred_2.pkl', './live/records/hundred_3.pkl',
               './live/records/hundred_4.pkl', './live/records/hundred_5.pkl']
    prepare(records)
