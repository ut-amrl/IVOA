#!/bin/bash

python ../train_multi_class_model.py \
 --data_path "/media/ssd2/datasets"\
"/introspective_failure_detection_noMedFilt_colored" \
 --meta_data_path  "/media/ssd2/datasets"\
"/introspective_failure_detection_noMedFilt_colored"  \
 --model_save_dir \
"/media/ssd2/results/IVOA/tmp_testing/"\
 --snapshot_save_dir \
"/media/ssd2/results/IVOA/tmp_testing/snapshot/"\
 --model_name "test_train_model"\
 --use_color_images true  \
 --online_patch_extraction true  \
 --patch_crop_size 100 \
 --fn_sample_weight_coeff 0.1 \
 --train_set "train_7" \
 --validation_set "valid_6"


