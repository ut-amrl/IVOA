#!/bin/bash

python ../visualize_inference_results.py \
 --data_path "/data/CAML/IVOA_CRA" \
 --model_dir \
"/data/CAML/IVOA_CRA/models/snapshot/"\
"cra_small_train_model_best_model_084.pt" \
--save_dir "/data/CAML/IVOA_CRA/visualization/" \
--use_multi_gpu_model false  \
--use_gpu true  \
--use_color_images true  \
--patch_crop_size 100 \
--test_set "test1"

