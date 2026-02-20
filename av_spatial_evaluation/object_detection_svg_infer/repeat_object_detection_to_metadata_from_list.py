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
    model = init_detector(config_file, checkpoint_file, device="cpu")  # or device="cuda:0" if mmdet-matched GPU

    thresh_conf = 0.3

    os.makedirs(pred_dir, exist_ok=True)

    for video_path in tqdm.tqdm(list_video_path):
        cap = cv2.VideoCapture(video_path)
        df = pd.DataFrame(columns=["frame", "category", "x0", "y0", "x1", "y1"])

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # assert frame_count == 20, "We expect the frame count is 20"
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps == 4, "We expect the fps is 4"

        for frame_gen in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_gen)
            ret, img = cap.read()

            result = inference_detector(model, img)
            array_labels = result.pred_instances.labels.cpu().detach().numpy()
            array_scores = result.pred_instances.scores.cpu().detach().numpy()
            array_bboxes = result.pred_instances.bboxes.cpu().detach().numpy()

            category = 0  # 0: person
            array_scores_category = array_scores[array_labels == category]
            array_bboxes_category = array_bboxes[array_labels == category]

            for i in range(len(array_scores_category)):
                if array_scores_category[i] > thresh_conf:
                    df.loc[len(df.index)] = [frame_gen, category,
                                             array_bboxes_category[i][0],
                                             array_bboxes_category[i][1],
                                             array_bboxes_category[i][2],
                                             array_bboxes_category[i][3]]

        if not df.empty:
            df = df.sort_values("frame")
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
