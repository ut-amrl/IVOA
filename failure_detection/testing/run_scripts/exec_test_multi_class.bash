#!/bin/bash

python ../test_multi_class_model.py \
 --data_path "/data/CAML/IVOA_CRA/" \
 --meta_data_path  "/data/CAML/IVOA_CRA/"  \
 --model_dir "/data/CAML/IVOA_CRA/models/snapshot/cra_full_train_model_unlocked_continued_best_model_016.pt" \
--save_dir "/data/CAML/IVOA_CRA/evaluation_multi_class_uncertainty/" \
--result_file_name  evaluation_multi_class_uncertainty  \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--calc_uncertainty true  \
--online_patch_extraction true  \
--patch_crop_size 50 \
--test_set "test_1"


