import cv2
import glob 
import os
import argparse
from torchvision.io import read_video, write_jpeg

  
def extract(SOURCE_PATH, DEST_PATH):
    # print('Named explicitly:') 
    SOURCE_PATH=SOURCE_PATH+"/*.mp4"

    for video_path in glob.glob(SOURCE_PATH): 
        name = video_path.split("/")[-1].replace(".mp4","")
        print(">>> name: ", name)
        # vidcap = cv2.VideoCapture(video_path)
        frames, _, _ = read_video((video_path), output_format="TCHW")
        for i in range(len(frames)):
            write_jpeg(frames[i], DEST_PATH+ f"/image_%s_%d.jpg" % (name, i))
        # success, image = vidcap.read()
        # count = 1
        # while success:
        #     cv2.imwrite(DEST_PATH+"/image_%s_%d.jpg" % (name, count), image)    
        #     success, image = vidcap.read()
        #     print('Saved image ', count)
        #     count += 1

if __name__=='__main__':
    SOURCE_PATH="/mnt/data2/simon/datasets/SAVGBench_Dataset_Development/video_dev/"
    DEST_PATH="/mnt/data2/simon/datasets/SAVGBench_Dataset_Development/image_dev/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        required=True, 
        help="source video files",
    )

    parser.add_argument(
        "-t",
        "--target_path",
        type=str,
        required=True, 
        help="path to config .yaml file",
    )
    args = parser.parse_args()
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
    
    extract(args.source_path, args.target_path)
        