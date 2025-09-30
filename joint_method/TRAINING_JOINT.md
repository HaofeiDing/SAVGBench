# Joint Baseline Method - Stereo MM-Diffusion

We adopt and modify the official PyTorch implementation of the paper [MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation](https://arxiv.org/abs/2212.09478), which is accepted by CVPR 2023.

Specifically, we modify the code to accommodate stereo audio for SAVGBench.
  
## Introduction

MM-Diffusion is the first joint audio-video generation framework that brings engaging watching and listening experiences simultaneously, towards high-quality realistic videos.
MM-Diffusion consists of a sequential multi-modal U-Net.
Two subnets for audio and video learn to gradually generate aligned audio-video pairs from Gaussian noises. 

We expand the mono audio to stereo audio to work on SAVGBench in an unconditional manner.

### Overview

<img src="./fig/MM-UNet2.png" width=100%>
(*Taken from the MM-Diffusion paper)

### Visual

The generated audio-video examples for the model trained on SAVGBench:

Please see the videos: "./fig/sample_1.mp4", "./fig/sample_2.mp4", "./fig/sample_3.mp4", "./fig/sample_4.mp4"

## Requirements and dependencies

* python 3.8 (recommend to use [Anaconda](https://www.anaconda.com/))
* pytorch >= 1.11.0
```
conda create -n mmdiffusion python=3.8
conda activate mmdiffusion
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch-nightly -c nvidia
conda install mpi4py
pip install -r requirements_training_joint.txt
```

## Dataset

Download the development set of the SAVGBench dataset: [Zenodo](https://zenodo.org/records/17139882)

## Train

```
# Training the base model
bash ssh_scripts/multimodal_train.sh

# Training an upsampler from 64x64 -> 256x256 
# First extract videos into frames for SR training
bash ssh_scripts/image_sr_train.sh

** You might need to set the PATH in multimodal_train.sh and image_sr_train.sh
```

## Test
```
# Testing the trained model
bash ssh_scripts/multimodal_sample_sr.sh

** You might need to set the PATH in multimodal_sample_sr.sh
```
  
## Acknowledgement

Thanks to [MM-Diffusion Codebase](https://github.com/researchmm/MM-Diffusion) for providing the base for our code.
