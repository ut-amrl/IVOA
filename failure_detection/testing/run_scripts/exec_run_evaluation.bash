#!/bin/bash

python ../run_evaluation.py \
 --model_type alex_multi \
 --data_path "/media/ssd2/datasets/AirSim_IVOA/airsim_ivoa" \
 --meta_data_path  "/media/ssd2/datasets/AirSim_IVOA/airsim_ivoa" \
 --model_dir \
"/media/ssd2/nn_models/IVOA/"\
"alex_multi_7_color_noMedFilt_best_model.pt" \
--save_dir "/media/ssd2/results/IVOA/tmp_testing" \
--result_file_name airsim_ivoa_test2  \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--calc_uncertainty false  \
--patch_crop_size 100  \
--stride_size 20  \
--eval_bbox_size 20  \
--workers_num 1 \
--bagfile_id 0

