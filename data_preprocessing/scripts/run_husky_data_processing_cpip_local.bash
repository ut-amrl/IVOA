#!/bin/bash


export ROS_PACKAGE_PATH=`pwd`/..:$ROS_PACKAGE_PATH


SESSIONS=( 0 1 2 3 4 5 6 )

# CAM_EXTRINSICS_PATH="../util/husky_Jun_6_21_calib_left.yaml"
CAM_EXTRINSICS_PATH="../util/husky_Jun_6_21_calib_left_ds.yaml"

# SOURCE_DIR="/media/ssd2/datasets/Husky_CPIP/Tr0_v0"
SOURCE_DIR="/media/ssd2/datasets/Husky_CPIP/Tr0_v0_ds"


PATCH_SIZE="30"
OUTPUT_DATASET_DIR="/media/ssd2/datasets/Husky_IVOA/CPIP_Tr0_v1_p30_ds/"

DEBUG="false"
VISUALIZATION="true"

MAX_RANGE="8.0"
MARGIN_WIDTH="20" # "50"
POSITIVE_HEIGHT_OBS_THRESH="0.1"
NEGATIVE_HEIGHT_OBS_THRESH="0.3"
# IMAGE_WIDTH="1224"
# IMAGE_HEIGHT="1024"
IMAGE_WIDTH="612"
IMAGE_HEIGHT="512"
DISTANCE_ERR_THRESH="1.0"
RELATIVE_DISTANCE_ERR_THRESH="0.2"

for session in ${SESSIONS[@]}; do
  printf -v SESSION_NUM_STR '%05d' "$session"
  
  echo "*********************************"
  echo "Generating Dataset Session $SESSION_NUM_STR"
  echo "*********************************"

  ../bin/airsim_data_processing \
  --session_num=$session \
  --source_dir=$SOURCE_DIR/$SESSION_NUM_STR \
  --cam_extrinsics_path=$CAM_EXTRINSICS_PATH \
  --output_dataset_dir=$OUTPUT_DATASET_DIR \
  --max_range=$MAX_RANGE \
  --min_range="0.1" \
  --debug=$DEBUG \
  --visualization=$VISUALIZATION \
  --pred_depth_fmt="pfm" \
  --pred_depth_folder="pred_depth" \
  --ground_plane_height="0.0" \
  --margin_width=$MARGIN_WIDTH \
  --patch_size=$PATCH_SIZE \
  --positive_height_obs_thresh=$POSITIVE_HEIGHT_OBS_THRESH \
  --negative_height_obs_thresh=$NEGATIVE_HEIGHT_OBS_THRESH \
  --image_width=$IMAGE_WIDTH \
  --image_height=$IMAGE_HEIGHT \
  --distance_err_thresh=$DISTANCE_ERR_THRESH \
  --relative_distance_err_thresh=$RELATIVE_DISTANCE_ERR_THRESH

done

