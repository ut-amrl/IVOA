#!/bin/bash


export ROS_PACKAGE_PATH=`pwd`/..:$ROS_PACKAGE_PATH


SESSIONS="1001"


SOURCE_DIR="/home/srabiee/data/AirSim_CPIP/CPIP_Tr0"

OUTPUT_DIR="/home/srabiee/data/IVOA/CPIP_TR0/evaluation"


GT_CAM_CALIBRATION="../util/gt_depth_calibration_cpip.yaml"
ML_CAM_CALIBRATION="../util/stereo_image_proc_calib_cpip.yaml"

DEBUG="true"
VISUALIZATION="true"

MAX_RANGES=(20)

for session in $SESSIONS; do
  for range in "${MAX_RANGES[@]}"; do
    printf -v SESSION_NUM_STR '%05d' "$session"
    printf -v RANGE_NUM_STR '%d' "$range"

    echo "*********************************"
    echo "Evaluating ML Tool on Session $SESSION_NUM_STR for Max Range 
$RANGE_NUM_STR"
    echo "*********************************"

    printf -v RANGE_DIR '%s/range_%d' "$OUTPUT_DIR" "$range"
    mkdir -p $RANGE_DIR

    TRAJECTORY_PATH=$SOURCE_DIR/$SESSION_NUM_STR/"state.txt"

    ../bin/mltool_evaluation \
    --session_num=$session \
    --source_dir=$SOURCE_DIR/$SESSION_NUM_STR \
    --gt_cam_cal_path=$GT_CAM_CALIBRATION \
    --ml_cam_cal_path=$ML_CAM_CALIBRATION \
    --output_dir=$RANGE_DIR \
    --trajectory_path=$TRAJECTORY_PATH \
    --max_range=$range \
    --debug=$DEBUG \
    --visualization=$VISUALIZATION \
    --pred_depth_fmt="pfm" \
    --pred_depth_folder="pred_depth" \
    --ground_plane_height="0.0"
  done
done


