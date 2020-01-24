#!/bin/bash

# For embeddings of size 2048

python ../extract_embedding_r.py \
 --data_path "/hdd/datasets"\
"/introspective_failure_detection_noMedFilt_colored" \
 --meta_data_path  "/hdd/datasets/"\
"/introspective_failure_detection_noMedFilt_colored"  \
 --model_dir \
"/hdd/nn_models/failure_detection/"\
"alex_multi_7_color_noMedFilt_best_model.pt" \
--save_dir "/hdd/results/introspective_failure_detection/"\
"alex_multi_7_color_noMedFilt/embeddings_2048/" \
--result_file_name  alex_multi_7  \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--calc_uncertainty false  \
--online_patch_extraction true  \
--patch_crop_size 100 \
--test_set newIndoor



