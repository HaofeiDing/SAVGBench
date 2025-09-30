# Copyright 2023 Sony Group Corporation.

import numpy as np
import random
import math
import pandas as pd


def select_time(train_wav_length, wav, fs):
    center = random.randrange(round(0        + train_wav_length / 2 * fs),
                              round(len(wav) - train_wav_length / 2 * fs))
    start = center - round(train_wav_length / 2 * fs)

    return start

def change_category_id(line_prev, category_id_config):
    category_id_prev = line_prev[1]
    category_id_new = category_id_config["{}".format(int(category_id_prev))]
    line_new = line_prev
    line_new[1] = category_id_new
    return line_new


def make_time_array(real_csv, category_id_config=None):
    if category_id_config is not None:
        time_list_prev = pd.read_csv(real_csv, header=None).values.tolist()
        time_list_new = [change_category_id(each_line, category_id_config) for each_line in time_list_prev]
        return np.array(time_list_new)
    else:
        return pd.read_csv(real_csv, header=None).values


def add_label_each_frame(label_cat, label_azi, time_array4frame_event, start_frame):
    category = int(time_array4frame_event[1])
    azi = int(time_array4frame_event[2])

    label_cat[category, start_frame: start_frame + 10] = 1
    label_azi[category, start_frame: start_frame + 10] = azi

    return label_cat, label_azi


class Label4StereoSELD(object):
    def __init__(self, num_cat, num_frame_wide):
        super().__init__()
        self._label_wide_cat_0 = np.zeros([num_cat, num_frame_wide])
        self._label_wide_azi_0 = np.zeros([num_cat, num_frame_wide])

    def add_label_each_frame(self, list_time_array4frame_event, start_frame):
        self._label_wide_cat_0, self._label_wide_azi_0 = add_label_each_frame(
            self._label_wide_cat_0, self._label_wide_azi_0, list_time_array4frame_event[0], start_frame)

    def concat(self, index_diff, num_frame):
        label_cat = self._label_wide_cat_0[np.newaxis, :, index_diff: index_diff + num_frame]
        label_azi = self._label_wide_azi_0[np.newaxis, :, index_diff: index_diff + num_frame]
        return label_cat, label_azi


def get_label(time_array, start_sec, train_wav_length, class_num):
    num_cat = class_num
    num_frame = round(train_wav_length * 100) + 1

    end_sec = start_sec + train_wav_length

    index_diff = int(math.modf(start_sec * 10)[0] * 10)  # get second decimal place
    num_frame_wide = (int(np.ceil(end_sec * 10)) - int(np.floor(start_sec * 10)) + 1) * 10
    # "+ 1" is buffer for numerical error, such as index_diff=3 and num_frame_wide=130

    label_class = Label4StereoSELD(num_cat, int(num_frame_wide))

    for index, frame in enumerate(range(int(np.floor(start_sec * 10)), int(np.ceil(end_sec * 10)))):
        time_array4frame = time_array[time_array[:, 0] == frame]  # (0, 5) shape is ok
        sorted_time_array4frame = time_array4frame[np.argsort(time_array4frame[:, 1])]

        list_time_array4frame_event = []
        for i in range(len(sorted_time_array4frame)):
            list_time_array4frame_event.append(sorted_time_array4frame[i])
            if i == len(sorted_time_array4frame) - 1:  # if the last
                label_class.add_label_each_frame(list_time_array4frame_event, index * 10)
                list_time_array4frame_event = []

    label_cat, label_azi = label_class.concat(int(index_diff), num_frame)

    return label_cat, label_azi
