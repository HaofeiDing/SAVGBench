import tqdm
import os
import pandas as pd
import numpy as np
import sys


def is_sound_near_to_object(x_seld, df_od_around):
    azi_thresh = 40

    x0_seld = x_seld - azi_thresh
    x1_seld = x_seld + azi_thresh

    for row in df_od_around.values:
        frame_od, category_od, x0_od, y0_od, x1_od, y1_od = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])

        if (x1_od >= x0_seld) and (x0_od <= x1_seld):
            return True

    return False


def main(list_txt_video_path, pred_seld_dir, pred_od_dir):
    list_video_path = [x[0] for x in pd.read_table(list_txt_video_path, header=None).values.tolist()]

    fps_seld = 10
    fps_od = 4

    allow_err_frame_od = 1

    num_TP = 0
    num_FN = 0

    for video_path in tqdm.tqdm(list_video_path):
        csv_file = os.path.basename(video_path).replace(".mp4", ".csv")
        pred_seld_path = os.path.join(pred_seld_dir, csv_file)
        pred_od_path = os.path.join(pred_od_dir, csv_file)

        df_seld = pd.read_csv(pred_seld_path, sep=",", header=None) if os.path.getsize(pred_seld_path) > 0 else pd.DataFrame([])
        df_od = pd.read_csv(pred_od_path, sep=",", header=None) if os.path.getsize(pred_od_path) > 0 else pd.DataFrame([])

        for row_seld in df_seld.values:
            frame_seld, category_seld, x_seld = int(row_seld[0]), int(row_seld[1]), int(row_seld[2])

            if df_od.empty:
                df_od_around = pd.DataFrame([]) # the same as df_od, both are empty
            else:
                frame_od = int(np.round(frame_seld / fps_seld * fps_od))
                
                # Dynamic clamping based on available OD frames
                if not df_od.empty:
                    max_available_frame = df_od[0].max()
                    if frame_od > max_available_frame:
                         frame_od = max_available_frame
                
                # Remove assertion or make it dynamic
                # assert (frame_od >= 0) and (frame_od < 20), "We expect frame_seld 0 and 49 is the first and last frames"
                df_od_around = df_od[(df_od[0] >= frame_od - allow_err_frame_od) & (df_od[0] <= frame_od + allow_err_frame_od)]

            if is_sound_near_to_object(x_seld, df_od_around):
                num_TP += 1
            else:
                num_FN += 1

    print(num_TP, num_FN, num_TP + num_FN)
    print(num_TP / (num_TP + num_FN))

    each_component = "AV spatial each component: num_TP: {}, num_FN: {}, num_TP + num_FN: {} \n".format(num_TP, num_FN, num_TP + num_FN) 
    av_final_score = "AV_final_score: {} \n".format(num_TP / (num_TP + num_FN))
    with open("results.out", "a") as myfile:
        myfile.write(each_component)
        myfile.write(av_final_score)


if __name__ == "__main__":
   args = sys.argv
   assert len(args) == 4, "We expect three args: python compute_spatialAValign_metrics.py LIST_TXT_VIDEO_PATH PRED_SELD_DIR PRED_OD_DIR"
   # python compute_spatialAValign_metrics.py ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/list_path_video.txt ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/pred_seld_20240614042355_0040000 ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/pred_yolox_tiny_mmdetection
   # python compute_spatialAValign_metrics.py ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/list_path_video_200x400.txt ~/avgen/stereo_seld/data/model_monitor/20240614042355/pred_TMP4VAL ~/avgen/object_detection_svg/data/model_monitor/yolox_tiny_mmdetection/pred

   list_txt_video_path = args[1]
   pred_seld_dir = args[2]
   pred_od_dir = args[3]

   main(list_txt_video_path, pred_seld_dir, pred_od_dir)
