import tqdm
import pandas as pd
import numpy as np
import moviepy.editor
import os
import random
import sys


def visualize_positions(video_path, new_video_path,
                        shape_type_video,
                        gt_path, pred_seld_path, pred_od_path,
                        fps_gt, fps_seld, fps_od):
    video_metadata = moviepy.editor.VideoFileClip(video_path)
    duration_sv_sec = np.round(video_metadata.duration)  # 5.0
    video_metadata = video_metadata.set_duration(duration_sv_sec)
    video_metadata = moviepy.editor.CompositeVideoClip([video_metadata])
    line_weight = 5
    height_video = 144
    azi_thresh = 40
    if shape_type_video == "256x256_pad5656x00":  # pad
        pad_top_video = 56

    if os.path.getsize(pred_seld_path) > 0:
        rows_seld = pd.read_csv(pred_seld_path, sep=",", header=None).values
    else:
        rows_seld = []
    list_color_cat = [np.array((0, 0, 0)), np.array((255, 255, 255))]
    for row in rows_seld:
        frame, category, x_seld = int(row[0]), int(row[1]), int(row[2])
        color_cat = list_color_cat[category]  # tmp only two categories: 0 and 1
        start_sec = float(frame / fps_seld)
        duration = 1 / fps_seld
        box_top = moviepy.editor.ColorClip(size=(azi_thresh * 2, line_weight), color=color_cat)
        box_top = box_top.set_position((x_seld - azi_thresh, pad_top_video)).set_start(start_sec).set_duration(duration)
        box_bottom = moviepy.editor.ColorClip(size=(azi_thresh * 2, line_weight), color=color_cat)
        box_bottom = box_bottom.set_position((x_seld - azi_thresh, pad_top_video + height_video - line_weight)).set_start(start_sec).set_duration(duration)
        box_left = moviepy.editor.ColorClip(size=(line_weight, height_video), color=color_cat)
        box_left = box_left.set_position((x_seld - azi_thresh, pad_top_video)).set_start(start_sec).set_duration(duration)
        box_right = moviepy.editor.ColorClip(size=(line_weight, height_video), color=color_cat)
        box_right = box_right.set_position((x_seld + azi_thresh - line_weight, pad_top_video)).set_start(start_sec).set_duration(duration)
        video_metadata = moviepy.editor.CompositeVideoClip([video_metadata, box_top, box_bottom, box_left, box_right])

    if os.path.getsize(pred_od_path) > 0:
        rows_od = pd.read_csv(pred_od_path, sep=",", header=None).values
    else:
        rows_od = []
    for row in rows_od:
        frame, category, x0_od, y0_od, x1_od, y1_od = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])
        color_cat = np.array((0, 255, 0)) # tmp only one category: 0
        start_sec = float(frame / fps_od)
        duration = 1 / fps_od
        box_top = moviepy.editor.ColorClip(size=(x1_od - x0_od, line_weight), color=color_cat)
        box_top = box_top.set_position((x0_od, y0_od)).set_start(start_sec).set_duration(duration)
        box_bottom = moviepy.editor.ColorClip(size=(x1_od - x0_od, line_weight), color=color_cat)
        box_bottom = box_bottom.set_position((x0_od, y1_od - line_weight)).set_start(start_sec).set_duration(duration)
        box_left = moviepy.editor.ColorClip(size=(line_weight, y1_od - y0_od), color=color_cat)
        box_left = box_left.set_position((x0_od, y0_od)).set_start(start_sec).set_duration(duration)
        box_right = moviepy.editor.ColorClip(size=(line_weight, y1_od - y0_od), color=color_cat)
        box_right = box_right.set_position((x1_od - line_weight, y0_od)).set_start(start_sec).set_duration(duration)
        video_metadata = moviepy.editor.CompositeVideoClip([video_metadata, box_top, box_bottom, box_left, box_right])

    if os.path.exists(gt_path):
        assert os.path.getsize(gt_path) > 0, "We expect rows_gt is not empty since the video should contain one or more onscreen sounds"
        rows_gt = pd.read_csv(gt_path, sep=",", header=None).values
    else:
        rows_gt = pd.DataFrame([]).values
    for row in rows_gt:
        frame, category, x, y = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        text = moviepy.editor.TextClip("{}".format(category), transparent=True, fontsize=10, bg_color="aqua")
        start_sec = float(frame / fps_gt)
        duration = 1 / fps_gt
        text = text.set_position((x, y + pad_top_video)).set_start(start_sec).set_duration(duration)
        video_metadata = moviepy.editor.CompositeVideoClip([video_metadata, text])

    audio = moviepy.editor.AudioFileClip(video_path, fps=16000).set_duration(duration_sv_sec)
    audio = moviepy.editor.CompositeAudioClip([audio])
    video_metadata.audio = audio
    video_metadata.write_videofile(new_video_path)


def main(list_txt_video_path, shape_type_video, pred_seld_dir, pred_od_dir):
    list_video_path = [x[0] for x in pd.read_table(list_txt_video_path, header=None).values.tolist()]
    if len(list_video_path) <= 10:
        list_video_path_random = list_video_path
    else:
        list_video_path_random = random.sample(list_video_path, k=10)

    fps_gt = 10
    fps_seld = 10
    fps_od = 4

    for video_path in tqdm.tqdm(list_video_path_random):
        new_video_path = video_path.replace("video", "video_label_gt_seld_od")
        os.makedirs(os.path.dirname(new_video_path), exist_ok=True)

        gt_dir = os.path.dirname(video_path).replace("video", "metadata")
        csv_file = os.path.basename(video_path).replace(".mp4", ".csv")
        gt_path = os.path.join(gt_dir, csv_file)
        pred_seld_path = os.path.join(pred_seld_dir, csv_file)
        pred_od_path = os.path.join(pred_od_dir, csv_file)

        # visualize positions with 3 csvs
        visualize_positions(video_path, new_video_path,
                            shape_type_video,
                            gt_path, pred_seld_path, pred_od_path,
                            fps_gt, fps_seld, fps_od)


if __name__ == "__main__":
   args = sys.argv
   assert len(args) == 5, "We expect two args: python visualize_estimated_position.py LIST_TXT_VIDEO_PATH SHAPE_TYPE_VIDEO PRED_SELD_DIR PRED_OD_DIR"
   # python visualize_estimated_position.py ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/list_path_video.txt 256x256_pad6464x00 ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/pred_seld_20240614042355_0040000 ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/pred_yolox_tiny_mmdetection
   # python visualize_estimated_position.py ~/avgen/dataset/STARSS23_PlanarStereo_generated_20240626/list_path_video_200x400.txt 200x400 ~/avgen/stereo_seld/data/model_monitor/20240614042355/pred_TMP4VAL ~/avgen/object_detection_svg/data/model_monitor/yolox_tiny_mmdetection/pred

   list_txt_video_path = args[1]
   shape_type_video = args[2]
   pred_seld_dir = args[3]
   pred_od_dir = args[4]

   main(list_txt_video_path, shape_type_video, pred_seld_dir, pred_od_dir)
