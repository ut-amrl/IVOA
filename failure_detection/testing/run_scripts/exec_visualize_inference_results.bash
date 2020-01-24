#!/bin/bash

python ../visualize_inference_results.py \
 --data_path "/media/ssd2/datasets/AirSim_IVOA/airsim_ivoa" \
 --model_dir \
"/media/ssd2/nn_models/IVOA/"\
"alex_multi_7_color_noMedFilt_best_model.pt" \
--save_dir "/media/ssd2/results/IVOA/tmp_testing" \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--patch_crop_size 100 \
--test_set "test1"

