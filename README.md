# SAVGBench: Benchmarking Spatially Aligned Audio-Video Generation

This repository includes the brief introduction, dataset link, and source code for "SAVGBench: Benchmarking Spatially Aligned Audio-Video Generation."

## Abstract

This work addresses the lack of multimodal generative models capable of producing high-quality videos with spatially aligned audio.
While recent advancements in generative models have been successful in video generation, they often overlook the spatial alignment between audio and visuals, which is essential for immersive experiences.
To tackle this problem, we establish a new research direction in benchmarking the Spatially Aligned Audio-Video Generation (SAVG) task.
We introduce a spatially aligned audio-visual dataset, whose audio and video data are curated based on whether sound events are onscreen or not.
We also propose a new alignment metric that aims to evaluate the spatial alignment between audio and video.
Then, using the dataset and metric, we benchmark two types of baseline methods: one is based on a joint audio-video generation model, and the other is a two-stage method that combines a video generation model and a video-to-audio generation model.
Our experimental results demonstrate that gaps exist between the baseline methods and the ground truth in terms of video and audio quality, as well as spatial alignment between the two modalities.

## Generated Videos

https://github.com/user-attachments/assets/2038cab3-06f5-47d2-826d-8cf6461d9be3

## SAVGBench Dataset

The SAVGBench dataset can be downloaded from [Zenodo](https://zenodo.org/records/17139882).
We use the development set for training and the evaluation set for evaluation.

https://github.com/user-attachments/assets/edcd51f4-c167-418c-9d4e-e4ed94ba8021

## Code

Under preparation.
