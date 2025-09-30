# Copyright 2023 Sony Group Corporation.

import argparse
import os


def file_path(string):
    if os.path.isfile(string):
        return string
    elif string == 'None':
        return None
    else:
        raise FileNotFoundError(string)


def dir_path_no_error(string):
    if string == 'None':
        return None
    else:
        return string


def dir_path(string):
    if os.path.isdir(string):
        return string
    elif string == 'None':
        return None
    else:
        raise NotADirectoryError(string)


def model_monitor_path(string):
    if string == './data/model_monitor':  # default is OK even if not a dir
        return string
    else:
        return dir_path(string)


def get_args():
    parser = argparse.ArgumentParser()

    # setup
    parser.add_argument('--train', '-train', action='store_true', help='Train.')
    parser.add_argument('--val', '-val', action='store_true', help='Val.')
    parser.add_argument('--eval', '-eval', action='store_true', help='Eval.')
    parser.add_argument('--infer', '-infer', action='store_true', help='Infer.')
    parser.add_argument('--quick-check', action='store_true', help='Quick check for new implementation.')
    parser.add_argument('--monitor-path', '-m', type=model_monitor_path, default='./data/model_monitor', help='Path monitoring logs saved.')
    parser.add_argument('--random-seed', '-rs', type=int, default=0, help='Seed number for random and np.random.')
    parser.add_argument('--train-wav-from', default='memory', choices=['memory', 'disk_wav', 'disk_npy'], help='Train wav from memory or disk.')
    parser.add_argument('--val-wav-from', default='disk', choices=['memory', 'disk'], help='Val wav from memory or disk.')
    parser.add_argument('--disk-config', type=file_path, default=None, help='Config file can be used to change disk of data.')
    parser.add_argument('--parallel-gpu', action='store_true', help='GPU parallel computation')
    parser.add_argument('--num-worker', type=int, default=0, help='Number of workers for torch.utils.data.DataLoader.')
    parser.add_argument('--pred-dir', type=dir_path_no_error, default=None, help='Pred directory can be set by argument.')
    # task
    parser.add_argument('--class-num', type=int, default=2, help='Total number of target classes, 13 is default for STARSS23.')
    parser.add_argument('--train-wav-txt', '-twt', type=file_path, default=None, help='Train wave file list text.')
    parser.add_argument('--val-wav-txt', '-valwt', type=file_path, default=None, help='Val wave file list text.')
    parser.add_argument('--eval-wav-txt', '-evalwt', type=file_path, default=None, help='Eval wave file list text.')
    parser.add_argument('--eval-model', '-em', type=file_path, default='av_spatial_evaluation/stereo_seld_infer/data/model_monitor/20240912162834/params_swa_20240912162834_0040000.pth', help='Eval model.')
    parser.add_argument('--category-id-config', type=file_path, default='av_spatial_evaluation/stereo_seld_infer/util/category_id_019to001.json', help='Config file can be used to change category id.')
    # net
    parser.add_argument('--net', '-n', default='cnn_mhsa_fc', choices=['cnn_mhsa_fc'], help='Neural network architecture.')
    parser.add_argument('--net-config', type=file_path, default='av_spatial_evaluation/stereo_seld_infer/net/net_small.json', help='Config file is required for net.')
    parser.add_argument('--act-azi', default='none', choices=['none', 'sigmoid', 'tanh'], help='Activation of azimuth output in net.')
    parser.add_argument('--range-azi', default='0to256_0to1', choices=['0to256_0to1', '0to256_-1to1'], help='Expected range of azimuth output from net.')
    # optimizer
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001)
    parser.add_argument('--weight-decay', '-w', type=float, default=0.000001, help='Weight decay factor of SGD update.')
    parser.add_argument('--max-iter', '-i', type=int, default=40000, help='Max iteration of training.')
    parser.add_argument('--model-save-interval', '-s', type=int, default=1000, help='The interval of saving model parameters.')
    parser.add_argument('--loss-azi', default='masked_mse', choices=['mse', 'masked_mse'], help='Loss type of azimuth.')
    # feature
    parser.add_argument('--sampling-frequency', '-fs', type=int, default=16000, help='Sampling frequency.')
    parser.add_argument('--feature', default='amp_phasediff', choices=['amp_phasediff'], help='Input audio feature type.')
    parser.add_argument('--fft-size', type=int, default=512, help='FFT size.')
    parser.add_argument('--stft-hop-size', type=int, default=160, help='STFT hop size.')
    parser.add_argument('--train-wav-length', type=float, default=1.27, help='Train wav length [seconds].')
    parser.add_argument('--eval-wav-hop-length', type=float, default=1.2, help='Eval wav hop length [seconds].')
    parser.add_argument('--feature-config', type=file_path, default='av_spatial_evaluation/stereo_seld_infer/feature/feature.json', help='Config file is required for feature.')
    # threshold
    parser.add_argument('--threshold-config', type=file_path, default='av_spatial_evaluation/stereo_seld_infer/util/threshold.json', help='Config file is required for threshold.')

    args = parser.parse_args()

    return args
