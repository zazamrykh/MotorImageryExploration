import os
import pickle
import random
from datetime import datetime
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import pywt
import torch
from pylsl import StreamInlet, resolve_stream

from src.functions import generate_mt_freq, visualize_mt
from src.params import event_duration, wavelet, device, frequencies_num
from src.wavelets.dwt1d import generate_int_psi_scales, cwt1d


class States(Enum):
    PREPARATION = 7
    CHILLING = 0
    LH = 1
    RH = 2
    PASSIVE = 3
    LL = 4
    RL = 6
    TONGUE = 5


def find_slope_and_shift(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


class Record:
    def __init__(self, channels, sampling_rate=125, preparation_time=5.0):
        self.start_time = datetime.now()
        self.data = {'marker': []}
        self.channels = channels  # List of channels in order
        for channel in channels:
            self.data[channel] = []
        self.prev_time = preparation_time
        self.prev_sample = [0 for _ in range(len(channels))]
        self.prev_index = -1
        self.sampling_rate = sampling_rate
        self.preparation_time = preparation_time
        self.beginning_time = None

    def add_data(self, sample, marker):
        self.data['marker'].append(marker)
        for i, channel in enumerate(self.channels):
            self.data[self.channels[i]].append(sample[i])

    def add_data_uniformly(self, sample, marker, time):
        for i in range(self.prev_index + 1, 1 + int((time - self.preparation_time) * self.sampling_rate)):
            t = i / self.sampling_rate
            self.data['marker'].append(marker)
            for j, channel in enumerate(self.channels):
                a, b = find_slope_and_shift(self.prev_time, self.prev_sample[j], time, sample[j])
                self.data[channel].append(a * t + b)
        self.prev_sample = sample
        self.prev_time = time
        self.prev_index = len(self.data['marker']) - 1

    def visualize(self, sample_num, channel_str):
        plt.plot(self.data[channel_str][-sample_num:-1])
        plt.title(channel_str)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.show()

    def save(self, filename):
        while os.path.exists(filename + '.pkl'):
            filename += '_'
        with open(filename + '.pkl', 'wb') as pickle_file:
            self.data['beginning_date'] = self.beginning_time
            pickle.dump(self.data, pickle_file)


def get_images(path_to_images):
    main_img = cv2.imread(path_to_images + 'main.bmp')
    lh_img = cv2.imread(path_to_images + 'lh.bmp')
    rh_img = cv2.imread(path_to_images + 'rh.bmp')
    passive_img = cv2.imread(path_to_images + 'passive.bmp')
    ll_img = cv2.imread(path_to_images + 'll.bmp')
    rl_img = cv2.imread(path_to_images + 'rl.bmp')
    tongue_img = cv2.imread(path_to_images + 'tongue.bmp')
    return main_img, lh_img, rh_img, passive_img, ll_img, rl_img, tongue_img


def print_time_left(t, preparing_time, time_flags):
    if time_flags[-1] or t > preparing_time:
        return
    for i in range(0, len(time_flags)):
        if not time_flags[i] and preparing_time - t < len(time_flags) - i:
            time_flags[i] = True
            print('Prepare... Experiments starts in', preparing_time - t)


def show_im_action(state, window_name, img):
    if state == States.LH:
        cv2.imshow(window_name, img)
    elif state == States.RH:
        cv2.imshow(window_name, img)
    elif state == States.PASSIVE:
        cv2.imshow(window_name, img)
    elif state == States.LL:
        cv2.imshow(window_name, img)
    elif state == States.RL:
        cv2.imshow(window_name, img)
    elif state == States.TONGUE:
        cv2.imshow(window_name, img)


class State:
    def __init__(self):
        self.state = States.PREPARATION
        self.chill_was_started = False
        self.state_start_time = None
        self.state_time = None

    def set_chill(self, t, chill_time, window_name, main_img):
        self.state = States.CHILLING
        self.state_start_time = t
        self.state_time = chill_time
        cv2.imshow(window_name, main_img)
        cv2.waitKey(1)

    def set_im_action(self, im_action_number, time, window_name, img):
        self.state = States(im_action_number)
        self.state_start_time = time
        self.state_time = event_duration
        show_im_action(self.state, window_name, img)
        cv2.waitKey(1)


def main():
    # first resolve an EEG stream on the lab network
    frequencies = generate_mt_freq(frequencies_num)
    scales = pywt.frequency2scale(wavelet, frequencies / 125)
    int_psi_scales = generate_int_psi_scales(scales, wavelet, device)

    print("looking for an EEG stream...")
    streams = resolve_stream("type", "EEG")

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    main_img, lh_img, rh_img, passive_img, ll_img, rl_img, tongue_img = get_images('./images/')
    imgs = {States.CHILLING: main_img, States.LH: lh_img, States.RH: rh_img, States.PASSIVE: passive_img,
            States.LL: ll_img, States.RL: rl_img, States.TONGUE: tongue_img}
    channels = ['O1', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4', 'O2']
    record = Record(channels)

    # class that contains all variables of states

    preparing_time = 5.0

    state = State()
    state.state = States.PREPARATION

    sampling_rate = 125

    first_pass_was = False
    beginning_timestamp = 0.0
    time_flags = [False, False, False, False, False]
    window_name = 'Experiment'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    i = 0
    motor_imagery_index = -1
    total_imagery_number = 100
    while True:
        sample, timestamp = inlet.pull_sample()

        if not first_pass_was:
            first_pass_was = True
            beginning_timestamp = timestamp
            beginning_time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
            record.beginning_time = beginning_time
            print("Experiment beginning time:", beginning_time)

        t = timestamp - beginning_timestamp  # Time in seconds

        if state.state == States.PREPARATION:
            if t > preparing_time:
                chill_time = 1.5 + random.random()  # [1.5  2.5]
                state.set_chill(t, chill_time, window_name, main_img)
                record.add_data(sample, 0)
            print_time_left(t, preparing_time, time_flags)
            record.add_data(sample, 0)

        elif state.state == States.CHILLING:
            if t - state.state_start_time > state.state_time:
                imagery_action_number = random.randint(1, 6)
                motor_imagery_index += 1
                if motor_imagery_index == total_imagery_number:
                    record.save('./records/hundred_14')
                    return
                print('Imagination number:', motor_imagery_index)
                state.set_im_action(imagery_action_number, t, window_name, imgs[States(imagery_action_number)])
                record.add_data(sample, imagery_action_number)
            else:
                record.add_data(sample, 0)

        else:
            record.add_data(sample, 0)
            if t - state.state_start_time > state.state_time:
                chill_time = 1.5 + random.random()
                state.set_chill(t, chill_time, window_name, main_img)
                if motor_imagery_index % 10 == 0:
                    record.visualize(sampling_rate, 'C3')
                # wt_result = cwt1d(torch.tensor(record.data['C3'][-sampling_rate:-1]).to(device), scales, int_psi_scales, out_dtype='real', device=device)
                # visualize_mt(wt_result.to('cpu'), frequencies)

        i += 1

if __name__ == "__main__":
    main()
