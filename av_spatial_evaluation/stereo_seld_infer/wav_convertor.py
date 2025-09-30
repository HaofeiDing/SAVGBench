# Copyright 2023 Sony Group Corporation.

import numpy as np
import soundfile as sf
import pandas as pd
import json
import os

from feature.feature import SpectralFeature
from util.func_seld_data_loader import get_label, make_time_array

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # ignore librosa's "UserWarning: PySoundFile failed. Trying audioread instead."
import librosa


class WavConvertor(object):
    def __init__(self, args):
        self._args = args
        with open(self._args.feature_config, 'r') as f:
            self._feature_config = json.load(f)
        self._fs = self._args.sampling_frequency
        self._frame_len_in_train = round(self._args.train_wav_length * self._fs / self._args.stft_hop_size) + 1
        self._hop_frame_len = round(self._args.eval_wav_hop_length * self._fs / self._args.stft_hop_size)
        if self._args.category_id_config is not None:
            with open(self._args.category_id_config, 'r') as f:
                self._category_id_config = json.load(f)
        else:
            self._category_id_config = None

    def wav_path2wav(self, wav_path):
        if os.path.splitext(wav_path) == ".wav":
            wav, _ = sf.read(wav_path, dtype='float32', always_2d=True)
        else:
            wav_librosa, _ = librosa.load(wav_path, sr=self._fs, mono=False)
            wav = wav_librosa.T
        wav_ch = wav.shape[1]
        if len(wav) % self._args.stft_hop_size != 0:
            wav = wav[0:-(len(wav) % self._args.stft_hop_size)]
        wav_pad = np.concatenate((np.zeros((self._args.fft_size - self._args.stft_hop_size, wav_ch), dtype='float32'), wav), axis=0)
        duration = len(wav) / self._fs
        return wav_pad, duration

    def wav2spec(self, wav_pad):
        spec_feature = SpectralFeature(wav=wav_pad,
                                       fft_size=self._args.fft_size,
                                       stft_hop_size=self._args.stft_hop_size,
                                       center=False,
                                       config=self._feature_config)
        if self._args.feature == 'amp_phasediff':
            spec = np.concatenate((spec_feature.amplitude(),
                                   spec_feature.phasediff()))

        pad_init = spec[:, :, -int(np.floor((self._frame_len_in_train - self._hop_frame_len) / 2)):]  # loop
        # pad_init = spec[:, :, :int(np.floor((self._frame_len_in_train - self._hop_frame_len) / 2))][:, :, ::-1]  # reflect
        pad_end = spec[:, :, :self._frame_len_in_train]  # loop
        # pad_end = spec[:, :, -self._frame_len_in_train:][:, :, ::-1]  # reflect
        spec_pad = np.concatenate((pad_init, spec, pad_end), axis=2)

        return spec_pad

    def wav_path2label(self, wav_path, duration):
        csv_path = wav_path.replace('audio', 'metadata').replace('video', 'metadata').replace('.wav', '.csv').replace('.mp4', '.csv')
        if self._args.eval_wav_txt and "evaltest" in self._args.eval_wav_txt:
            csv_array = np.array([[-1, -1, -1, -1]])  # dummy for evaltest
        elif self._args.eval_wav_txt and "list_path_video" in self._args.eval_wav_txt:
            csv_array = np.array([[-1, -1, -1, -1]]) # dummy for list_path_video
        else:
            csv_array = make_time_array(csv_path, self._category_id_config)
        label_cat, label_azi = get_label(csv_array, 0, duration, self._args.class_num)

        pad_cat_init = np.zeros((label_cat.shape[0], label_cat.shape[1],
                                 int(np.floor((self._frame_len_in_train - self._hop_frame_len) / 2))), dtype='float32')
        pad_cat_end = np.zeros((label_cat.shape[0], label_cat.shape[1],
                                self._frame_len_in_train), dtype='float32')
        label_cat_pad = np.concatenate((pad_cat_init, label_cat, pad_cat_end), axis=2)
        pad_azi_init = np.zeros((label_azi.shape[0], label_azi.shape[1],
                                 int(np.floor((self._frame_len_in_train - self._hop_frame_len) / 2))), dtype='float32')
        pad_azi_end = np.zeros((label_azi.shape[0], label_azi.shape[1],
                                self._frame_len_in_train), dtype='float32')
        label_azi_pad = np.concatenate((pad_azi_init, label_azi, pad_azi_end), axis=2)

        return label_cat_pad, label_azi_pad
