#!/bin/bash

python ../test_multi_class_model.py \
 --data_path "/media/ssd2/datasets/AirSim_IVOA/ivoa_dataset_testing2" \
 --meta_data_path  "/media/ssd2/datasets/AirSim_IVOA/ivoa_dataset_testing2" \
 --model_dir \
"/media/ssd2/results/IVOA/tmp_testing/"\
"monodepth_ivoa_last_model.pt" \
--save_dir "/media/ssd2/results/IVOA/initial_results" \
--result_file_name  airsim_ivoa_test2_raw  \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--calc_uncertainty false  \
--online_patch_extraction true  \
--patch_crop_size 100 \
--test_set "test1"


