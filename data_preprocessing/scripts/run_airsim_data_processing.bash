#!/bin/bash


export ROS_PACKAGE_PATH=`pwd`/..:$ROS_PACKAGE_PATH


SESSIONS="3"


CAM_EXTRINSICS_PATH="/home/srabiee/My_Repos/IVOA/data_preprocessing/"\
"util/Camera_Extrinsics.yaml"

SOURCE_DIR="/media/ssd2/datasets/AirSim_IVOA/initial_ml_tool"
OUTPUT_DATASET_DIR="/media/ssd2/datasets/AirSim_IVOA/airsim_ivoa_test/"


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

