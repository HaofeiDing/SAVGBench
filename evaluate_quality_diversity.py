import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
from av_quality_evaluation import dist_util, logger
from av_quality_evaluation.evaluator import eval_multimodal
from av_quality_evaluation.common import  delete_pkl


# command: mpiexec -n 4 python py_scripts/eval.py --devices 0,1,2,3

def main(
    ):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, default=" ", help="path to reference batch npz file")
    parser.add_argument("--fake_dir", type=str, default=" ", help="path to sample batch npz file")
    parser.add_argument("--output_dir", type=str, default=" ", help="" )
    parser.add_argument("--video_size", type=str, default="20,3,256,256", help="video size")
    parser.add_argument("--audio_size", type=str, default="2,80000", help="audio size")
    parser.add_argument("--sample_num", type=int, default=2048)
    parser.add_argument("--devices", type=str, default="G8")
    args = parser.parse_args()
    args.video_size = [int(i) for i in args.video_size.split(',')]
    args.audio_size = [int(i) for i in args.audio_size.split(',')]

    dist_util.setup_dist(args.devices)
    logger.configure(dir=args.output_dir, log_suffix="_val")

    metric = eval_multimodal(args.ref_dir, args.fake_dir, eval_num=args.sample_num, video_size=args.video_size, audio_size=args.audio_size)
    
    with open("results.out", "a") as myfile:
        myfile.write(str(metric)+"\n")
    # logger.log(f"metric:{metric}")
    delete_pkl(args.fake_dir)    
        
if __name__ == '__main__':
    main()

