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
MAX_RANGES=( 10 )

for session in $SESSIONS; do
  for range in "${MAX_RANGES[@]}"; do
    printf -v SESSION_NUM_STR '%04d' "$session"
    printf -v RANGE_NUM_STR '%d' "$range"

    echo "*********************************"
    echo "Evaluating ML Tool on Session $SESSION_NUM_STR for Max Range $RANGE_NUM_STR"
    echo "*********************************"

    printf -v RANGE_DIR '%s/session_%s_range_%d' "$OUTPUT_DIR" "$SESSION_NUM_STR" "$range"
    mkdir -p $RANGE_DIR

    ../bin/mltool_evaluation \
    --session_num=$session \
    --source_dir=$SOURCE_DIR \
    --cam_extrinsics_path=$CAM_EXTRINSICS_PATH \
    --output_dir=$RANGE_DIR \
    --trajectory_path=$TRAJECTORY_PATH \
    --max_range=$range
  done
done


