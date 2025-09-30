#!/bin/bash

# check with generated sounding videos
DATASET_DIR=/home/shimada/avgen/dataset/STARSS23_PlanarStereo_generated_20240626
LIST_TXT_VIDEO_PATH=$DATASET_DIR/list_path_video.txt
PRED_SELD_DIR=$DATASET_DIR/pred_seld_swa_20240912162834_0040000
PRED_OD_DIR=$DATASET_DIR/pred_yolox_tiny_mmdetection
SHAPE_TYPE_VIDEO=256x256_pad5656x00

# # check with original dataset
# DATASET_DIR=/home/shimada/avgen/dataset/STARSS23_PerspectiveStereo_filterhighpassampdb38_lenfrm50stpfrm5firfrm0thrfrm40stpdeg10firdeg0_interest019_add
# LIST_TXT_VIDEO_PATH=$DATASET_DIR/list_path_video_eval.txt
# PRED_SELD_DIR=$DATASET_DIR/pred_seld_swa_20240912162834_0040000
# PRED_OD_DIR=$DATASET_DIR/pred_yolox_tiny_mmdetection
# SHAPE_TYPE_VIDEO=256x256_pad5656x00

echo sound event localization and detection
python stereo_seld_infer/seld.py \
 -infer -evalwt $LIST_TXT_VIDEO_PATH --pred-dir $PRED_SELD_DIR;

echo object detection
python object_detection_svg_infer/repeat_object_detection_to_metadata_from_list.py \
 $LIST_TXT_VIDEO_PATH $PRED_OD_DIR;

echo spatial audiovisual alignment
python compute_spatialAValign_metrics.py \
 $LIST_TXT_VIDEO_PATH $PRED_SELD_DIR $PRED_OD_DIR;

echo visualization of estimated position
python visualize_estimated_position.py \
 $LIST_TXT_VIDEO_PATH $SHAPE_TYPE_VIDEO $PRED_SELD_DIR $PRED_OD_DIR;
