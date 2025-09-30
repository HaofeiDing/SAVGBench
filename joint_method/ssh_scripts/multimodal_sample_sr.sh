
MODEL_FLAGS="--cross_attention_resolutions 2,4,8  --cross_attention_windows 1,4,8 
--cross_attention_shift True  --video_attention_resolutions 2,4,8
--audio_attention_resolutions -1
--video_size 20,3,64,64 --audio_size 2,80000  --learn_sigma False --num_channels 128 
--num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --video_fps  4
--use_scale_shift_norm True"

SRMODEL_FLAGS="--sr_attention_resolutions 8,16,32  --large_size 256  
--small_size 64 --sr_learn_sigma True 
--sr_num_channels 192 --sr_num_heads 4 --sr_num_res_blocks 2 
--sr_resblock_updown True --use_fp16 False --sr_use_scale_shift_norm True"

# Modify --devices according your GPU number, generate 96 samples
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
--all_save_num 96 --save_type mp4  --devices  G8
--batch_size 2  --is_strict True --sample_fn dpm_solver++"

SR_DIFFUSION_FLAGS="--sr_diffusion_steps 1100  --sr_sample_fn ddim  --sr_timestep_respacing ddim25"

# Modify the following paths to your own paths
MULTIMODAL_MODEL_PATH="{YOUR_MODEL_PATH}/model330011.pt" ## MMDiffusion model
SR_MODEL_PATH="{YOUR_MODEL_PATH}/model_SR_mmdiff_120000.pt" ## Super resolution model
OUT_DIR="{YOUR_OUTPUT_PATH}"
REF_PATH="{YOUR_DATA_PATH}/SAVGBench_Dataset_Development/video_dev/" 

mkdir -p $OUT_DIR

NUM_GPUS=1

mpiexec -n $NUM_GPUS python3 multimodal_sample_sr.py  \
$MODEL_FLAGS $SRMODEL_FLAGS $DIFFUSION_FLAGS $SR_DIFFUSION_FLAGS --ref_path ${REF_PATH} \
--output_dir ${OUT_DIR} --multimodal_model_path ${MULTIMODAL_MODEL_PATH}  --sr_model_path ${SR_MODEL_PATH} 
