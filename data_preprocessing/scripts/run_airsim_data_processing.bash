#!/bin/bash


export ROS_PACKAGE_PATH=`pwd`/..:$ROS_PACKAGE_PATH

if [ -z "$1" ] || [ -z "$2" ]
then
  echo "USAGE: $0 SOURCE_DIR SESSION_NUM"
fi

CAM_EXTRINSICS_PATH="../util/gt_depth_calibration.yaml"

SOURCE_DIR=$1
OUTPUT_DATASET_DIR="/data/CAML/IVOA_CRA/"
SESSIONS=$2

for session in $SESSIONS; do
  printf -v SESSION_NUM_STR '%05d' "$session"

  echo "*********************************"
  echo "Generating Dataset Session $SESSION_NUM_STR"
  echo "*********************************"

  ..//bin/airsim_data_processing \
  --session_num=$session \
  --source_dir=$SOURCE_DIR \
  --cam_extrinsics_path=$CAM_EXTRINSICS_PATH \
  --output_dataset_dir=$OUTPUT_DATASET_DIR

done

