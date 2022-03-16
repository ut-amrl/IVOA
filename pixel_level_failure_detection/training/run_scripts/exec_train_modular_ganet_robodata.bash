#!/bin/bash

python \
../train_modular.py \
 --cfg "/robodata/srabiee/scratch/My_Repos/IVOA/pixel_level_failure_detection/config/airsim/airsimCity_mobilenetv2dialated-c1_deepsup_bin.yaml" \
 --gpus "0,1"
