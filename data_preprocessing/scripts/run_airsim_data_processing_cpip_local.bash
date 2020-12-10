#!/bin/bash


export ROS_PACKAGE_PATH=`pwd`/..:$ROS_PACKAGE_PATH


SESSIONS="1001"

CAM_EXTRINSICS_PATH="../util/gt_depth_calibration_cpip.yaml"


SOURCE_DIR="/home/srabiee/data/AirSim_CPIP/CPIP_Tr0"

OUTPUT_DATASET_DIR="/home/srabiee/data/IVOA/CPIP_TR0/"

DEBUG="false"
VISUALIZATION="false"

for session in $SESSIONS; do
  printf -v SESSION_NUM_STR '%05d' "$session"
  
  echo "*********************************"
  echo "Generating Dataset Session $SESSION_NUM_STR"
  echo "*********************************"

  ../bin/airsim_data_processing \
  --session_num=$session \
  --source_dir=$SOURCE_DIR/$SESSION_NUM_STR \
  --cam_extrinsics_path=$CAM_EXTRINSICS_PATH \
  --output_dataset_dir=$OUTPUT_DATASET_DIR \
  --max_range="20.0" \
  --debug=$DEBUG \
  --visualization=$VISUALIZATION \
  --pred_depth_fmt="pfm" \
  --pred_depth_folder="pred_depth" \
  --ground_plane_height="0.0" \
  --margin_width="50.0"
  
done

