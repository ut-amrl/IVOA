#!/bin/bash


export ROS_PACKAGE_PATH=`pwd`/..:$ROS_PACKAGE_PATH

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]
then
  echo "USAGE: $0 SOURCE_DIR OUT_DIR SESSION_NUM TRAJECTORY_PATH"
fi

CAM_EXTRINSICS_PATH="../util/Camera_Extrinsics.yaml"

SOURCE_DIR=$1
OUTPUT_DIR=$2
SESSIONS=$3
TRAJECTORY_PATH=$4

for session in $SESSIONS; do
  printf -v SESSION_NUM_STR '%05d' "$session"

  echo "*********************************"
  echo "Generating Dataset Session $SESSION_NUM_STR"
  echo "*********************************"

  ../bin/mltool_evaluation \
  --session_num=$session \
  --source_dir=$SOURCE_DIR \
  --cam_extrinsics_path=$CAM_EXTRINSICS_PATH \
  --output_dir=$OUTPUT_DIR \
  --trajectory_path=$TRAJECTORY_PATH

done


