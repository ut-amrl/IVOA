#!/bin/bash

python ../run_evaluation.py \
 --model_type alex_multi \
 --data_path "/data/CAML/IVOA_CRA/" \
 --meta_data_path  "/data/CAML/IVOA_CRA/" \
 --model_dir \
"/data/CAML/IVOA_CRA/models/snapshot/"\
"cra_full_train_model_unlocked_continued_best_model_011.pt" \
--save_dir "/data/CAML/IVOA_CRA/evaluation_unlocked_continued/" \
--result_file_name ivoa_cra_full_checkpoint_bag_3  \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--calc_uncertainty true  \
--patch_crop_size 50  \
--stride_size 20  \
--eval_bbox_size 20  \
--workers_num 1 \
--bagfile_id 3

