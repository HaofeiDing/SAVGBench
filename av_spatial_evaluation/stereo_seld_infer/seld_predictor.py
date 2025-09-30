# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn
import numpy as np
import json

from net.net_seld import create_net_seld
from seld_trainer import SEDDOA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # temporary for pandas
import pandas as pd


class SELDClassifier(object):
    def __init__(self, args):
        self._args = args

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._net = create_net_seld(self._args)
        if self._args.parallel_gpu:
            self._net = torch.nn.DataParallel(self._net)
        self._net.to(self._device)
        self._net.eval()
        checkpoint = torch.load(self._args.eval_model, map_location=lambda storage, loc: storage)
        if self._args.parallel_gpu:
            self._net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self._net.load_state_dict(checkpoint['model_state_dict'])

        self._criterion = SEDDOA(self._args.range_azi, self._args.loss_azi)

        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._frame_length = round(self._args.train_wav_length * fs / self._args.stft_hop_size) + 1
        self._hop_frame = round(self._args.eval_wav_hop_length * self._frame_per_sec)

    def set_input(self, spec_pad, label_pad):
        self._spec_pad = spec_pad
        self._label_cat_pad, self._label_azi_pad = label_pad

    def receive_input(self, time_array):
        features = np.zeros(tuple([self._args.batch_size]) + (self._spec_pad[:, :, :self._frame_length]).shape)
        labels_cat = np.zeros(tuple([self._args.batch_size]) + (self._label_cat_pad[:, :, :self._frame_length]).shape)
        labels_azi = np.zeros(tuple([self._args.batch_size]) + (self._label_azi_pad[:, :, :self._frame_length]).shape)

        for index, time in enumerate(time_array):
            frame_idx = int(time * self._frame_per_sec)
            features[index] = self._spec_pad[:, :, frame_idx: frame_idx + self._frame_length]
            labels_cat[index] = self._label_cat_pad[:, :, frame_idx: frame_idx + self._frame_length]
            labels_azi[index] = self._label_azi_pad[:, :, frame_idx: frame_idx + self._frame_length]

        self._input_a = torch.tensor(features, dtype=torch.float).to(self._device)
        self._label_cat = torch.tensor(labels_cat, dtype=torch.float).to(self._device)
        self._label_azi = torch.tensor(labels_azi, dtype=torch.float).to(self._device)

    def calc_output(self):
        output_net = self._net(self._input_a)
        self._output_cat = output_net[0]
        self._output_azi = output_net[1]

    def get_output(self):
        cut_frame = int(np.floor((self._frame_length - self._hop_frame) / 2))
        output_cat = self._output_cat.cpu().detach().numpy()
        output_azi = self._output_azi.cpu().detach().numpy()
        self._output_cat = 0  # for memory release
        self._output_azi = 0
        # only use output from cut [frame] to cut + hop [frame]
        return output_cat[:, :, :, cut_frame: cut_frame + self._hop_frame], output_azi[:, :, :, cut_frame: cut_frame + self._hop_frame]

    def get_loss(self):
        self._loss = self._criterion(self._output_cat, self._output_azi, self._label_cat, self._label_azi)
        loss = self._loss.cpu().detach().numpy()
        self._loss = 0  # for memory release
        return loss


class SELDDetector(object):
    def __init__(self, args):
        self._args = args
        with open(args.threshold_config, 'r') as f:
            threshold_config = json.load(f)
        self._thresh_bin = threshold_config['threshold_presence']

        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._hop_frame = round(self._args.eval_wav_hop_length * self._frame_per_sec)

    def set_duration(self, duration):
        self._duration = duration
        eval_wav_hop_length = self._args.eval_wav_hop_length
        if (self._duration % eval_wav_hop_length == 0) or (np.abs((self._duration % eval_wav_hop_length) - eval_wav_hop_length) < 1e-10):
            self._time_array = np.arange(0, self._duration + eval_wav_hop_length, eval_wav_hop_length)
        else:
            self._time_array = np.arange(0, self._duration, eval_wav_hop_length)

        self._df = pd.DataFrame(columns=["frame", "category", "azimuth"])
        self._minibatch_result_cat = np.zeros((
            len(self._time_array) + self._args.batch_size,
            1,  # track
            self._args.class_num,
            self._hop_frame))
        self._raw_output_array_cat = np.zeros((
            1,
            self._args.class_num,
            len(self._time_array) * self._hop_frame))
        self._minibatch_result_azi = np.zeros((
            len(self._time_array) + self._args.batch_size,
            1,
            self._args.class_num,
            self._hop_frame))
        self._raw_output_array_azi = np.zeros((
            1,
            self._args.class_num,
            len(self._time_array) * self._hop_frame))

    def get_time_array(self):
        return self._time_array

    def set_minibatch_result(self, index, result):
        result_cat, result_azi = result
        self._minibatch_result_cat[
            index * self._args.batch_size: (index + 1) * self._args.batch_size
        ] = result_cat
        self._minibatch_result_azi[
            index * self._args.batch_size: (index + 1) * self._args.batch_size
        ] = result_azi

    def minibatch_result2raw_output_array(self):
        array_len = (self._minibatch_result_cat.shape[0]) * self._hop_frame
        result_array_cat = np.zeros((self._raw_output_array_cat.shape[0], self._raw_output_array_cat.shape[1], array_len))
        for index, each_result in enumerate(self._minibatch_result_cat):
            result_array_cat[
                :, :, index * self._hop_frame: (index + 1) * self._hop_frame
            ] = each_result
        self._raw_output_array_cat = result_array_cat[:, :, : len(self._time_array) * self._hop_frame]
        result_array_azi = np.zeros((self._raw_output_array_azi.shape[0], self._raw_output_array_azi.shape[1], array_len))
        for index, each_result in enumerate(self._minibatch_result_azi):
            result_array_azi[
                :, :, index * self._hop_frame: (index + 1) * self._hop_frame
            ] = each_result
        self._raw_output_array_azi = result_array_azi[:, :, : len(self._time_array) * self._hop_frame]

    def detect(self, index, time):
        cat0 = self._raw_output_array_cat[0, :, index * self._hop_frame: (index + 1) * self._hop_frame]
        cat_sigmoid0 = self._numpy_sigmoid(cat0)
        azi0 = self._raw_output_array_azi[0, :, index * self._hop_frame: (index + 1) * self._hop_frame]

        self._frame_per_sec4csv = 10  # for csv setting
        hop_frame4csv = int(self._hop_frame / (self._frame_per_sec / self._frame_per_sec4csv))  # e.g., 12 [frame in csv]
        for csv_idx, frame in enumerate(range(int(time * self._frame_per_sec4csv), int(time * self._frame_per_sec4csv) + hop_frame4csv)):
            csv2net = int(self._frame_per_sec / self._frame_per_sec4csv)  # e.g., 100 [frame for net] / 10 [frame for csv]
            net_idx_start = csv_idx * csv2net
            net_idx_end = (csv_idx + 1) * csv2net

            cat_sigmoid_mean = np.mean(cat_sigmoid0[:, net_idx_start: net_idx_end], axis=1)
            azi_mean = np.sum(cat_sigmoid0[:, net_idx_start: net_idx_end] * azi0[:, net_idx_start: net_idx_end], axis=1)\
                  / np.sum(cat_sigmoid0[:, net_idx_start: net_idx_end], axis=1)

            for i in range(self._args.class_num):
                if cat_sigmoid_mean[i] > self._thresh_bin:
                    category = i
                    azimuth = self._change_range_azi(azi_mean[i])
                    self._append_df(frame, category, azimuth)

    def _change_range_azi(self, azi):
        if self._args.range_azi == "0to256_0to1":
            azimuth = azi * 256  # 0<=azi<1 -> 0<=azimuth<256
        elif self._args.range_azi == "0to256_-1to1":
            azimuth = (azi * 128) + 128  # -1<=azi<1 -> 0<=azimuth<256
        return azimuth

    def _numpy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # equal to np.array(torch.sigmoid(torch.tensor((x))))

    def _append_df(self, frame, cat, azi):
        self._df.loc[len(self._df.index)] = [frame, cat, azi]

    def save_df(self, pred_path):
        if not self._df.empty:
            self._df = self._df.sort_values("frame")
            self._df = self._df[self._df["frame"] < int(self._duration * self._frame_per_sec4csv)]  # cut frames after duration
        self._df.to_csv(pred_path, sep=',', index=False, header=False)
