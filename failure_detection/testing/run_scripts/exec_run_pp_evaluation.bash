#!/bin/bash

python ../run_pp_evaluation.py \
 --model_type alex_multi \
 --data_path "/media/ssd2/datasets/AirSim_IVOA/airsim_ivoa" \
 --meta_data_path  "/media/ssd2/datasets/AirSim_IVOA/airsim_ivoa" \
 --model_dir \
"/media/ssd2/nn_models/IVOA/"\
"alex_multi_7_color_noMedFilt_best_model.pt" \
--save_dir  "/media/ssd2/results/IVOA/tmp_testing" \
--pre_results_dir "/media/ssd2/results/IVOA/tmp_testing" \
--result_file_name ivoa_airsim_test_pp  \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--calc_uncertainty false  \
--patch_crop_size 100  \
--workers_num 2 \
--stride_size 20  \
--eval_bbox_size 20  \
--bagfile_id 0  

