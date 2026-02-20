import os
import tqdm
import cv2
import pandas as pd
import sys
from mmdet.apis import init_detector, inference_detector


def main(list_txt_video_path, pred_dir):
    list_video_path = [x[0] for x in pd.read_table(list_txt_video_path, header=None).values.tolist()]

    config_file = "av_spatial_evaluation/object_detection_svg_infer/yolox_tiny_8x8_300e_coco.py"
    checkpoint_file = "av_spatial_evaluation/object_detection_svg_infer/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
    
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file missing: {os.path.abspath(config_file)}")
    if not os.path.exists(checkpoint_file):
        print(f"[ERROR] Checkpoint file missing: {os.path.abspath(checkpoint_file)}")

    # Detect device
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[OD] Initializing YOLOX on {device}...")
    
    try:
        model = init_detector(config_file, checkpoint_file, device=device)
        print(f"[OD] Model loaded successfully.")
    except Exception as e:
        print(f"[OD] FAILED to initialize detector: {e}")
        return

    thresh_conf = 0.3
    os.makedirs(pred_dir, exist_ok=True)

    for video_path in tqdm.tqdm(list_video_path):
        if not os.path.exists(video_path):
            print(f"[OD] ERROR: Video not found at {video_path}")
            continue
            
        cap = cv2.VideoCapture(video_path)
        df = pd.DataFrame(columns=["frame", "category", "x0", "y0", "x1", "y1"])

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[OD] Processing {os.path.basename(video_path)}: {frame_count} frames found.")

        for frame_gen in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_gen)
            ret, img = cap.read()
            if not ret or img is None:
                print(f"[OD]   Warning: Failed to read frame {frame_gen}")
                continue

            result = inference_detector(model, img)
            
            # Extract boxes
            instances = result.pred_instances
            array_labels = instances.labels.cpu().detach().numpy()
            array_scores = instances.scores.cpu().detach().numpy()
            array_bboxes = instances.bboxes.cpu().detach().numpy()

            # Detection targets:
            # The original benchmark was person-centric, but for general YouTube 
            # clips, we want to detect ANY object that could be making sound.
            # COCO has 80 classes; if we use a broad range, alignment is more robust.
            
            target_categories = None # None means detect all categories
            
            frame_found = 0
            for idx in range(len(array_labels)):
                cat = array_labels[idx]
                score = array_scores[idx]
                # If target_categories is None, detect all. Otherwise only in list.
                if (target_categories is None or cat in target_categories) and score > thresh_conf:
                    df.loc[len(df.index)] = [frame_gen, cat,
                                             array_bboxes[idx][0],
                                             array_bboxes[idx][1],
                                             array_bboxes[idx][2],
                                             array_bboxes[idx][3]]
                    frame_found += 1
            
            # if frame_found > 0:
            #    print(f"[OD]   Frame {frame_gen}: found {frame_found} boxes")

        if not df.empty:
            df = df.sort_values("frame")
            print(f"[OD] Saved {len(df)} total boxes for {os.path.basename(video_path)}")
        else:
            print(f"[OD] WARNING: 0 total boxes for {os.path.basename(video_path)}")
            
        pred_path = os.path.join(pred_dir,
                                 os.path.basename(video_path).replace(".mp4", ".csv"))
        df[["frame", "category"]] = df[["frame", "category"]].astype(int)
        df.to_csv(pred_path, sep=',', index=False, header=False)

if __name__ == "__main__":
   args = sys.argv
   assert len(args) == 3, "We expect two args: python repeat_object_detection_to_metadata_from_list.py LIST_TXT_VIDEO_PATH PRED_DIR"
   # python repeat_object_detection_to_metadata_from_list.py ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/list_path_video.txt ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/pred_yolox_tiny_mmdetection

   list_txt_video_path = args[1]
   pred_dir = args[2]

   main(list_txt_video_path, pred_dir)
