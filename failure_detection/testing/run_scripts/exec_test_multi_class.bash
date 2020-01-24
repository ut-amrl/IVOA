#!/bin/bash

python ../test_multi_class_model.py \
 --data_path "/media/srabiee/Elements/srabiee/datasets"\
"/introspective_failure_detection_noMedFilt_colored" \
 --meta_data_path  "/media/srabiee/Elements/srabiee/datasets"\
"/introspective_failure_detection_noMedFilt_colored"  \
 --model_dir \
"/media/ssd2/nn_models/IVOA/"\
"alex_multi_7_color_noMedFilt_best_model.pt" \
--save_dir "/media/ssd2/results/IVOA/tmp_testing" \
--result_file_name  bag29_30_newIndoor  \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--calc_uncertainty false  \
--online_patch_extraction true  \
--patch_crop_size 100 \
--test_set "test1"


