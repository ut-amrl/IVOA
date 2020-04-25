#!/bin/bash

python ../train_multi_class_model.py \
 --data_path "/data/CAML/IVOA_CRA/" \
 --meta_data_path  "/data/CAML/IVOA_CRA/"  \
 --model_save_dir "/data/CAML/IVOA_CRA/models/"\
 --snapshot_save_dir \
"/data/CAML/IVOA_CRA/models/snapshot/"\
 --model_name "cra_full_train_model_unlocked_continued"\
 --use_color_images true  \
 --online_patch_extraction true  \
 --patch_crop_size 50 \
 --fn_sample_weight_coeff 0.1 \
 --train_set "train_1" \
 --validation_set "valid_1"


