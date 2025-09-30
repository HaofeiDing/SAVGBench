
#################64 x 64 uncondition###########################################################
# 
# MODEL_FLAGS="--cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8
# --cross_attention_shift True --dropout 0.1 
# --video_attention_resolutions 2,4,8
# --audio_attention_resolutions -1
# --video_size 20,3,64,64 --audio_size 2,80000 --learn_sigma False --num_channels 128 --video_fps  4
# --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True 
# --use_scale_shift_norm True --num_workers 8 --channel_mult 1,2,3 " ### target resolution: | 256, 144 | 384,216 | 512, 288 
# # Modify --devices to your own GPU ID, the current setting is for 8 gpus 
# TRAIN_FLAGS="--lr 0.0002 --batch_size 4 
# --devices G8 --log_interval 100 --save_interval 10000 --use_db False " # --devices G8 --schedule_sampler loss-second-moment
# DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000 --save_type mp4 --sample_fn dpm_solver"  # dpm_solver++" 

# Modify the following pathes to your own paths
DATA_DIR="{YOUR_DATA_PATH}/SAVGBench_Dataset_Development/video_dev/" 
OUTPUT_DIR="{YOUR_OUTPUT_PATH}"
NUM_GPUS=8

#RESUME_CHECKPOINT="{YOUR_MODEL_PATH}/model100000.pt" 

# Modify --devices to your own GPU ID, the current setting is for 8 gpus 
mpiexec -n $NUM_GPUS  python3 py_scripts/multimodal_train.py \
 --cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8 \
 --cross_attention_shift True --dropout 0.1 \
 --video_attention_resolutions 2,4,8 --audio_attention_resolutions -1 \
 --video_size 20,3,64,64 --audio_size 2,80000 --learn_sigma False --num_channels 128 --video_fps  4 \
 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
 --use_scale_shift_norm True --num_workers 12 \
 --lr 0.0002 --batch_size 4 --noise_schedule linear --diffusion_steps 1000 --save_type mp4 --sample_fn dpm_solver \
 --devices G8 --log_interval 100 --save_interval 10000 --use_db False \
 --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} # --resume_checkpoint ${RESUME_CHECKPOINT}
