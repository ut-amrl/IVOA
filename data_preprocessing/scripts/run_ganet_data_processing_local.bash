#!/bin/bash


export ROS_PACKAGE_PATH=`pwd`/..:$ROS_PACKAGE_PATH


SESSIONS=( 1008 )

# CAM_EXTRINSICS_PATH="../util/airsim_city_wb_calibration_ds.yaml"
CAM_EXTRINSICS_PATH="../util/airsim_city_wb_calibration.yaml"


SOURCE_DIR="/media/ssd2/datasets/AirSim_IVSLAM/cityenv_wb"
SOURCE_DIR_PREDICTIONS="/media/ssd2/results/ipr/depth_prediction/ganet_deep_airsim_sample4_00/model_ensemble_epoch_32_e034/cityenv_wb"
OUTPUT_DATASET_DIR="/media/ssd2/IVOA/GANET/ganet_deep_airsim_sample4_00_epoch_32_e034_v0_p70_rg30_pixelWise_errThresh1.0_0.2/"

DEBUG="false"
VISUALIZATION="false"
PIXEL_WISE_MODE="true"

export ROS_NAMESPACE="ivoa_r30_p70"
# -----------------

IMAGE_WIDTH="960"
IMAGE_HEIGHT="600"
# IMAGE_WIDTH="480"
# IMAGE_HEIGHT="300"

MAX_RANGE="30.0" # "30.0", "35.0"
MIN_RANGE="1.0"
PATCH_SIZE="70"
DISTANCE_ERR_THRESH="1.0" # "1.0" "1.0"
RELATIVE_DISTANCE_ERR_THRESH="0.2" # "0.2" "0.3"
MIN_PATCH_INFO_RATIO="0.1"
MIN_ERR_PIXEL_RATIO="0.1"
MARGIN_WIDTH="20" 
POSITIVE_HEIGHT_OBS_THRESH="0.3" 
NEGATIVE_HEIGHT_OBS_THRESH="0.3"
GROUND_PLANE_HEIGHT="0.0"


mkdir -p $OUTPUT_DATASET_DIR
for session in ${SESSIONS[@]}; do
  printf -v SESSION_NUM_STR '%05d' "$session"
  
  echo "*********************************"
  echo "Generating Dataset Session $SESSION_NUM_STR"
  echo "*********************************"

  ../bin/airsim_data_processing \
  --session_num=$session \
  --source_dir=$SOURCE_DIR/$SESSION_NUM_STR \
  --source_dir_predictions=$SOURCE_DIR_PREDICTIONS/$SESSION_NUM_STR \
  --cam_extrinsics_path=$CAM_EXTRINSICS_PATH \
  --output_dataset_dir=$OUTPUT_DATASET_DIR \
  --max_range=$MAX_RANGE \
  --min_range=$MIN_RANGE \
  --debug=$DEBUG \
  --visualization=$VISUALIZATION \
  --pred_depth_fmt="pfm" \
  --pred_depth_folder="img_depth_pred" \
  --ground_plane_height=$GROUND_PLANE_HEIGHT \
  --margin_width=$MARGIN_WIDTH \
  --patch_size=$PATCH_SIZE \
  --positive_height_obs_thresh=$POSITIVE_HEIGHT_OBS_THRESH \
  --negative_height_obs_thresh=$NEGATIVE_HEIGHT_OBS_THRESH \
  --image_width=$IMAGE_WIDTH \
  --image_height=$IMAGE_HEIGHT \
  --distance_err_thresh=$DISTANCE_ERR_THRESH \
  --relative_distance_err_thresh=$RELATIVE_DISTANCE_ERR_THRESH \
  --flip_depth_images=true \
  --generate_labels_in_pixel_wise_mode=$PIXEL_WISE_MODE \
  --min_patch_info_ratio=$MIN_PATCH_INFO_RATIO \
  --min_err_pixel_ratio=$MIN_ERR_PIXEL_RATIO 

done

