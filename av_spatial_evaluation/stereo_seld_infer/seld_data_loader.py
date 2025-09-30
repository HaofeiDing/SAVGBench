# Copyright 2023 Sony Group Corporation.

import numpy as np
import pandas as pd
import soundfile as sf
import random
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import os
import librosa

from util.func_seld_data_loader import select_time, get_label, make_time_array
from feature.feature import SpectralFeature


def create_data_loader(args):
    data_set = SELDDataSet(args)
    return DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)


class SELDDataSet(Dataset):
    def __init__(self, args):
        # wav on RAM
        self._args = args
        if self._args.category_id_config is not None:
            with open(self._args.category_id_config, 'r') as f:
                self._category_id_config = json.load(f)
        else:
            self._category_id_config = None

        self._train_wav_dict = {}
        self._time_array_dict = {}
        if self._args.quick_check:
            self._wav_path_list = pd.read_table(self._args.train_wav_txt, header=None).values.tolist()[::2000]
        else:
            self._wav_path_list = pd.read_table(self._args.train_wav_txt, header=None).values.tolist()
        if self._args.disk_config is not None:
            with open(self._args.disk_config, 'r') as f:
                disk_config = json.load(f)
            dir_old = disk_config["dir_old"]
            dir_new = disk_config["dir_new"]
            self._train_wav_path_list = [x[0].replace(dir_old, dir_new) for x in self._wav_path_list]
        else:
            self._train_wav_path_list = [x[0] for x in self._wav_path_list]
        for train_wav_path in tqdm.tqdm(self._train_wav_path_list, desc='[Train initial setup]'):
            if self._args.train_wav_from == "memory":
                if os.path.splitext(train_wav_path) == ".wav":
                    self._train_wav_dict[train_wav_path] = sf.read(train_wav_path, dtype='float32', always_2d=True)
                else:
                    wav_librosa, fs = librosa.load(train_wav_path, sr=self._args.sampling_frequency, mono=False)
                    self._train_wav_dict[train_wav_path] = (wav_librosa.T, fs)
            elif self._args.train_wav_from in ["disk_wav", "disk_npy"]:
                self._train_wav_dict[train_wav_path] = None
            real_csv = train_wav_path.replace('audio', 'metadata').replace('video', 'metadata').replace('.wav', '.csv').replace('.mp4', '.csv')
            self._time_array_dict[train_wav_path] = make_time_array(real_csv, self._category_id_config)

        with open(self._args.feature_config, 'r') as f:
            self._feature_config = json.load(f)

    def __len__(self):
        return self._args.batch_size * self._args.max_iter  # e.g., 32 * 40000

    def __getitem__(self, idx):  # idx is dummy
        input_a, label_cat, label_azi, name_data = self._data_seld()
        return input_a, label_cat, label_azi, name_data

    def _data_seld(self):
        path, time_array, wav, fs, start = self._choice_wav(self._train_wav_dict)
        input_wav = wav[start: start + round(self._args.train_wav_length * fs)]
        input_spec = self._wav2spec(input_wav)

        label_cat, label_azi = get_label(time_array, start / fs, self._args.train_wav_length, self._args.class_num)
        label_cat_float = label_cat.astype(np.float32)
        label_azi_float = label_azi.astype(np.float32)

        start_sec = start / fs

        return input_spec, label_cat_float, label_azi_float, '{}_{}'.format(path, start_sec)

    def _choice_wav(self, train_wav_dict):
        path, wav_fs = random.choice(list(train_wav_dict.items()))
        time_array = self._time_array_dict[path]
        if self._args.train_wav_from == "memory":
            wav, fs = wav_fs
        elif self._args.train_wav_from == "disk_wav":
            if os.path.splitext(path) == ".wav":
                wav, fs = sf.read(path, dtype='float32', always_2d=True)
            else:
                wav_librosa, fs = librosa.load(path, sr=self._args.sampling_frequency, mono=False)
                wav = wav_librosa.T
        elif self._args.train_wav_from == "disk_npy":
            wav = np.load(path.replace(".wav", ".npy").replace(".mp4", ".npy"))
            fs = self._args.sampling_frequency
        start = select_time(self._args.train_wav_length, wav, fs)
        return path, time_array, wav, fs, start

    def _wav2spec(self, input_wav):
        spec_feature = SpectralFeature(wav=input_wav,
                                       fft_size=self._args.fft_size,
                                       stft_hop_size=self._args.stft_hop_size,
                                       center=True,
                                       config=self._feature_config)

        if self._args.feature == 'amp_phasediff':
            input_spec = np.concatenate((spec_feature.amplitude(),
                                         spec_feature.phasediff()))

        return input_spec
