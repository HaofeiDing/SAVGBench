# Copyright 2023 Sony Group Corporation.

import os
import codecs
import json

from dcase2022_task3_seld_metrics import parameters, cls_compute_seld_results


def all_seld_eval(args, pred_directory, result_path=None):
    if args.eval:
        with open(args.eval_wav_txt) as f:
            wav_file_list = [s.strip() for s in f.readlines()]
        wav_dir = os.path.dirname(wav_file_list[0])
    elif args.val:
        with open(args.val_wav_txt) as f:
            wav_file_list = [s.strip() for s in f.readlines()]
        wav_dir = os.path.dirname(wav_file_list[0])
    ref_desc_files = wav_dir.replace("audio", "metadata").replace("video", "metadata")
    pred_output_format_files = pred_directory

    params = parameters.get_params()
    params['unique_classes'] = args.class_num
    if args.category_id_config is not None:
        with open(args.category_id_config, 'r') as f:
            category_id_config = json.load(f)
    else:
        category_id_config = None
    score_obj = cls_compute_seld_results.ComputeSELDResults(params, ref_files_folder=os.path.dirname(ref_desc_files),
                                                            category_id_config=category_id_config)
    er20, f20, le, lr, seld_err, classwise_test_scr, other_scores = score_obj.get_SELD_Results(pred_output_format_files)
    er20_d, er20_i, er20_s, pre, rec, lf, lp, classwise_other_results = other_scores

    print('SELD scores')
    print('All\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP')
    print('All\t{:0.3f}\t{:0.3f}\t{:0.2f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(
        er20, f20, le, lr, seld_err, er20_d, er20_i, er20_s, pre, rec, lf, lp))
    if params['average'] == 'macro':
        print('Class-wise results')
        print('Class\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                cls_cnt,
                classwise_test_scr[0][cls_cnt],
                classwise_test_scr[1][cls_cnt],
                classwise_test_scr[2][cls_cnt],
                classwise_test_scr[3][cls_cnt],
                classwise_test_scr[4][cls_cnt],
                classwise_other_results[0][cls_cnt],
                classwise_other_results[1][cls_cnt],
                classwise_other_results[2][cls_cnt],
                classwise_other_results[3][cls_cnt],
                classwise_other_results[4][cls_cnt],
                classwise_other_results[5][cls_cnt],
                classwise_other_results[6][cls_cnt]))

    if result_path is not None:
        print('SELD scores',
              file=codecs.open(result_path, 'w', 'utf-8'))
        print('All\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP',
              file=codecs.open(result_path, 'a', 'utf-8'))
        print('All\t{:0.3f}\t{:0.3f}\t{:0.2f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(
            er20, f20, le, lr, seld_err, er20_d, er20_i, er20_s, pre, rec, lf, lp),
            file=codecs.open(result_path, 'a', 'utf-8'))
        if params['average'] == 'macro':
            print('Class-wise results',
                  file=codecs.open(result_path, 'a', 'utf-8'))
            print('Class\tER\tF\tLE\tLR\tSELD\tER_D\tER_I\tER_S\tP\tR\tLF\tLP',
                  file=codecs.open(result_path, 'a', 'utf-8'))
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
                    cls_cnt,
                    classwise_test_scr[0][cls_cnt],
                    classwise_test_scr[1][cls_cnt],
                    classwise_test_scr[2][cls_cnt],
                    classwise_test_scr[3][cls_cnt],
                    classwise_test_scr[4][cls_cnt],
                    classwise_other_results[0][cls_cnt],
                    classwise_other_results[1][cls_cnt],
                    classwise_other_results[2][cls_cnt],
                    classwise_other_results[3][cls_cnt],
                    classwise_other_results[4][cls_cnt],
                    classwise_other_results[5][cls_cnt],
                    classwise_other_results[6][cls_cnt]),
                    file=codecs.open(result_path, 'a', 'utf-8'))

    return er20, f20, le, lr, seld_err, other_scores
