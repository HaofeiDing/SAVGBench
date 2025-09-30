# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

from net.net_util import interpolate


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)
    with open(args.net_config, 'r') as f:
        net_config = json.load(f)
    if args.act_azi == "sigmoid":
        assert args.range_azi == "0to256_0to1", "When you set act_azi sigmoid, please set range_azi 0to256_0to1"
    elif args.act_azi == "tanh":
        assert args.range_azi == "0to256_-1to1", "When you set act_azi tanh, please set range_azi 0to256_-1to1"
    if args.net == 'cnn_mhsa_fc':
        Net = CNN_MHSA_FC(class_num=args.class_num,
                          in_channels=feature_config[args.feature]["ch"],
                          in_channels_sed=feature_config[args.feature]["ch_sed"],
                          net_config=net_config,
                          act_azi=args.act_azi)

    return Net


class CNN_MHSA_FC(nn.Module):
    def __init__(self, class_num, in_channels, in_channels_sed, net_config, act_azi, interp_ratio=4):
        super().__init__()

        self.pe_enable = False  # True | False

        self.out_channels1 = net_config["conv"]["out_channels1"]
        self.out_channels2 = net_config["conv"]["out_channels2"]
        self.out_channels3 = net_config["conv"]["out_channels3"]

        self.d_model = net_config["transformer"]["d_model"]
        self.d_ff = net_config["transformer"]["d_ff"]
        self.n_heads = net_config["transformer"]["n_heads"]
        self.n_layers = net_config["transformer"]["n_layers"]

        self.class_num = class_num
        self.in_channels = in_channels
        self.in_channels_sed = in_channels_sed
        self.act_azi = act_azi
        self.interp_ratio = interp_ratio

        self.sed_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=self.in_channels_sed, out_channels=self.out_channels1),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels1, out_channels=self.out_channels2),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels2, out_channels=self.out_channels3),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.sed_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels3, out_channels=self.d_model),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.doa_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=self.in_channels, out_channels=self.out_channels1),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels1, out_channels=self.out_channels2),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels2, out_channels=self.out_channels3),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.doa_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=self.out_channels3, out_channels=self.d_model),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.stitch1 = nn.Parameter(torch.FloatTensor(self.out_channels1, 2, 2).uniform_(0.1, 0.9))
        self.stitch2 = nn.Parameter(torch.FloatTensor(self.out_channels2, 2, 2).uniform_(0.1, 0.9))
        self.stitch3 = nn.Parameter(torch.FloatTensor(self.out_channels3, 2, 2).uniform_(0.1, 0.9))

        if self.pe_enable:
            self.pe = PositionalEncoding(pos_len=100, d_model=self.d_model, pe_type='t', dropout=0.0)
        self.sed_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2), num_layers=self.n_layers)
        self.doa_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff, dropout=0.2), num_layers=self.n_layers)

        self.fc_sed_track1 = nn.Linear(self.d_model, self.class_num, bias=True)
        self.fc_doa_track1 = nn.Linear(self.d_model, self.class_num, bias=True)
        self.final_act_sed = nn.Sequential()  # nn.Sigmoid()
        if self.act_azi == "none":
            self.final_act_doa = nn.Sequential()
        elif self.act_azi == "sigmoid":
            self.final_act_doa = nn.Sigmoid()
        elif self.act_azi == "tanh":
            self.final_act_doa = nn.Tanh()

    def forward(self, x_a):
        x_a = x_a.transpose(2, 3)
        b, c, t, f = x_a.size()  # (N, C, T, F); N = batch_size, C = in_channels, T = time_frames, F = freq_bins

        x_sed = x_a[:, :self.in_channels_sed]
        x_doa = x_a

        # cnn
        x_sed = self.sed_conv_block1(x_sed)
        x_doa = self.doa_conv_block1(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch1[:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch1[:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch1[:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch1[:, 1, 1], x_doa)
        x_sed = self.sed_conv_block2(x_sed)
        x_doa = self.doa_conv_block2(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch2[:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch2[:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch2[:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch2[:, 1, 1], x_doa)
        x_sed = self.sed_conv_block3(x_sed)
        x_doa = self.doa_conv_block3(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch3[:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch3[:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch3[:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch3[:, 1, 1], x_doa)
        x_sed = self.sed_conv_block4(x_sed)
        x_doa = self.doa_conv_block4(x_doa)
        x_sed = x_sed.mean(dim=3)  # (N, C, T)
        x_doa = x_doa.mean(dim=3)  # (N, C, T)

        # transformer
        if self.pe_enable:
            x_sed = self.pe(x_sed)
        if self.pe_enable:
            x_doa = self.pe(x_doa)
        x_sed = x_sed.permute(2, 0, 1)  # (T, N, C)
        x_doa = x_doa.permute(2, 0, 1)  # (T, N, C)
        x_sed_1 = self.sed_trans_track1(x_sed).transpose(0, 1)  # (N, T, C)
        x_doa_1 = self.doa_trans_track1(x_doa).transpose(0, 1)  # (N, T, C)

        # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))  # (N, T, C)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))  # (N, T, C)

        # interpolate
        x_sed_1 = interpolate(x_sed_1, self.interp_ratio)
        x_sed_1 = x_sed_1.transpose(1, 2)  # (N, C, T)
        x_doa_1 = interpolate(x_doa_1, self.interp_ratio)
        x_doa_1 = x_doa_1.transpose(1, 2)  # (N, C, T)

        x_sed = torch.unsqueeze(x_sed_1, dim=1)  # (N, 1, class_num, T)
        x_doa = torch.unsqueeze(x_doa_1, dim=1)  # (N, 1, class_num, T)

        return x_sed, x_doa


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 dilation=1, bias=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        x = self.double_conv(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()

        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)
