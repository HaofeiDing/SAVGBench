#!/bin/bash
### $1: output video path  $2 reference video path  $3 number of gpus

OUT_DIR=$1  # "/mnt/data2/simon/outputs/MM-Diffusion-5s4fps/samples/multimodal-sample-sr/dpm_solver/model170000.pt_backup/"
REF_PATH=$2  # "/mnt/data2/simon/datasets/SAVGBench_Dataset_Evaluation/video_eval/"
NUM_GPUS=$3
NUM_SAMPLES=96

SAMPLE_DIR=$OUT_DIR  
OUTPUT_DIR=$OUT_DIR  

rm -r $OUT_DIR

./joint_method/run_baseline.sh $OUT_DIR $REF_PATH $NUM_GPUS $NUM_SAMPLES

OUT_DIR=$OUT_DIR/sr_mp4/

echo $OUT_DIR

rm ./results.out

ls ${OUT_DIR}/*.mp4  > ${OUT_DIR}/list_path_video.txt

LIST_TXT_VIDEO_PATH=${OUT_DIR}/list_path_video.txt
PRED_SELD_DIR=${OUT_DIR}/pred_seld_swa_20240912162834_0040000
PRED_OD_DIR=${OUT_DIR}/pred_yolox_tiny_mmdetection

echo sound event localization and detection
python av_spatial_evaluation/stereo_seld_infer/seld.py \
 -infer -evalwt $LIST_TXT_VIDEO_PATH --pred-dir $PRED_SELD_DIR;

echo object detection
python av_spatial_evaluation/object_detection_svg_infer/repeat_object_detection_to_metadata_from_list.py \
 $LIST_TXT_VIDEO_PATH $PRED_OD_DIR;

echo spatial audiovisual alignment
python av_spatial_evaluation/compute_spatialAValign_metrics.py \
 $LIST_TXT_VIDEO_PATH $PRED_SELD_DIR $PRED_OD_DIR;

echo evaluate quality
mpiexec -n 1 python evaluate_quality_diversity.py --devices 0 --sample_num ${NUM_SAMPLES} --ref_dir ${REF_PATH} --fake_dir ${SAMPLE_DIR} --output_dir ${OUTPUT_DIR}

echo evaluate AV-Align
python3 av_quality_evaluation/av_align_metric.py --input_dir $OUT_DIR
