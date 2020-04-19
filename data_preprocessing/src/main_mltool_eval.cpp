// Copyright 2020 srabiee@cs.utexas.edu, kvsikand@cs.utexas.edu
// Department of Computer Sciences,
// University of Texas at Austin
//
//
// This software is free: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License Version 3,
// as published by the Free Software Foundation.
//
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// Version 3 in the file COPYING that came with this distribution.
// If not, see <http://www.gnu.org/licenses/>.
// ========================================================================

#include <ros/ros.h>
#include <ros/console.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <iomanip>
#include <jsoncpp/json/json.h>
#include "yaml-cpp/yaml.h"


#include "io_access.h"
#include "depth2pointcloud.h"
#include"cnpy.h"

using cv::Mat;
using namespace std;
using namespace IVOA;
using std::vector;
using cv::Point;


// Command line flags flag
DEFINE_int32(session_num, 0, "Session number to identify the generated "
                             "portion of the dataset.");
DEFINE_string(source_dir, "", "Path to the base directory of the source " 
                              "dataset.");
DEFINE_string(cam_extrinsics_path, "", "Path to the file containing the "
                                       "left camera calibration file.");
DEFINE_string(output_dir, "", "Path to save the generated results. ");
DECLARE_bool(help);
DECLARE_bool(helpshort);


// Parameters
const bool kVisualization = false;


// Objects that are further than max_range_ away from the agent will not be
// considered for obstacle avoidance
const float kMaxRange = 20.0; // meters
const float kMinObstacleHeight = 0.3; // meters
const float kMaxObstacleHeight = 2.0; // meters

const cv::Size kImageSize(960, 600);

// Checks if all required command line arguments have been set
void CheckCommandLineArgs(char** argv) {
  vector<string> required_args = {"session_num",
                                  "source_dir",
                                  "cam_extrinsics_path",
                                  "output_dir"};
  
  for (const string& arg_name:required_args) {
    bool flag_not_set =   
          gflags::GetCommandLineFlagInfoOrDie(arg_name.c_str()).is_default;
    if (flag_not_set) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "main_mltool_eval");
      LOG(FATAL) << arg_name <<  " was not set." << endl;
    }
  }
}



int main(int argc, char **argv) {
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;   // INFO level logging.
  FLAGS_colorlogtostderr = 1;  // Colored logging.
  FLAGS_logtostderr = true;    // Don't log to disk
  
  
  string usage("This program detects and keeps track of failures in "
          "obstacle avoidance for a traveresed trajectory by the robot "
          "given the reference depth estimates as well as the predicted "
          "depth images by an ml model: \n");

  usage += string(argv[0]) + " <argument1> <argument2> ...";
  gflags::SetUsageMessage(usage);

  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_help) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "main_mltool_eval");
    return 0;
  }
  CheckCommandLineArgs(argv);
  
  ros::init(argc, argv, "mltool_evaluation");
  ros::NodeHandle nh;
 
  
  ros::Publisher point_cloud_publisher_gt =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/gt_pointcloud", 1);
  ros::Publisher point_cloud_publisher_pred =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/pred_pointcloud", 1);
  ros::Publisher point_cloud_publisher_gt_filt =
      nh.advertise<sensor_msgs::PointCloud2>(
          "/ivoa/gt_pointcloud_filtered", 1);
  ros::Publisher point_cloud_publisher_pred_filt =
      nh.advertise<sensor_msgs::PointCloud2>(
          "/ivoa/pred_pointcloud_filtered", 1);
  
  
  // Using only one instance of Depth2Pointcloud since the depth and left RGB
  // camera share the same extrinsics and intrinsics in our AirSim setup 
  Depth2Pointcloud depth_img_converter;
  depth_img_converter.LoadCameraCalibration(FLAGS_cam_extrinsics_path);
 

  string gt_depth_dir = FLAGS_source_dir + "/img_depth/";
  string pred_depth_dir = FLAGS_source_dir + "/img_left/";
  string left_img_dir = FLAGS_source_dir + "/img_left/";
 
  // Read the prefix of all avaiable datapoints
  vector<int> filename_prefixes;
  GetFileNamePrefixes(gt_depth_dir,
                      &filename_prefixes);
  std::sort(filename_prefixes.begin(), filename_prefixes.end());
  
  // TODO: load the trajectory file
  
  
  int count = 0;
  for (const int &i : filename_prefixes) {
    
    stringstream ss;
    ss << setfill('0') << setw(10) << i;
    string prefix = ss.str();
    string gt_depth_path = gt_depth_dir + prefix + ".pfm";
    string pred_depth_path = pred_depth_dir + prefix + "_disp.npy";
    string left_img_path = left_img_dir + prefix + ".png";

    
    // Read the left cam image
    Mat left_img = cv::imread(left_img_path,CV_LOAD_IMAGE_UNCHANGED);
    

    // Read the Ground truth depth image
    PFM pfm_rw;
    float * depth_data = 
                  pfm_rw.read_pfm<float>(gt_depth_path);
    cv::Mat depth_img_gt = cv::Mat(pfm_rw.getHeight(), 
                                pfm_rw.getWidth(), 
                                CV_32F, 
                                depth_data);
    cv::flip(depth_img_gt, depth_img_gt, 0);
    

    // Read the predicted depth image
    cnpy::NpyArray arr = cnpy::npy_load(pred_depth_path);
    float* loaded_data = arr.data<float>();
    cv::Mat depth_img_pred = cv::Mat(arr.shape[0],
                                     arr.shape[1],
                                     CV_32F,
                                     loaded_data);
  
    // TODO: Revisit how to resize the depth prediction. Maybe scale down the 
    // ground truth depth to match the predicted depth image.
    
    // Resize and linearly interpolate the predicted depth image so that it 
    // is the same size as the ground truth depth image
    cv::resize(depth_img_pred,depth_img_pred,depth_img_gt.size());
    
    // Ground truth and predicted point clouds
    sensor_msgs::PointCloud2 pt_cloud_gt;
    sensor_msgs::PointCloud2 pt_cloud_pred;
    if (!depth_img_converter.GeneratePointcloud(depth_img_gt, &pt_cloud_gt) ||
      !depth_img_converter.GeneratePointcloud(depth_img_pred, &pt_cloud_pred)){
      LOG(FATAL) << "Point cloud generation failed. "
                 << "No camera calibration was available.";
    }
    
    // Publish the original point clouds
    if (kVisualization) {
      point_cloud_publisher_gt.publish(pt_cloud_gt);
      point_cloud_publisher_pred.publish(pt_cloud_pred);
    }
    
    // Filter point clouds by height and range
    pt_cloud_gt = depth_img_converter.FilterPointCloudByHeightAndDistance(
                                                            pt_cloud_gt,
                                                            kMinObstacleHeight,
                                                            kMaxObstacleHeight,
                                                            0,
                                                            kMaxRange);
    
    pt_cloud_pred = depth_img_converter.FilterPointCloudByHeightAndDistance(
                                                            pt_cloud_pred,
                                                            kMinObstacleHeight,
                                                            kMaxObstacleHeight,
                                                            0,
                                                            kMaxRange);
    
    // Publish the filtered point clouds
    if (kVisualization) {
      point_cloud_publisher_gt_filt.publish(pt_cloud_gt);
      point_cloud_publisher_pred_filt.publish(pt_cloud_pred);
    }

    
    // TODO: Voxelize the pointclouds and do collision checking against both
    // the ground truth and predicted point cloud
    
    
    // TODO: Project the failure points to the image for visualization purposes
    
    // TODO: Keep track of the failures instances across time
    

    if (kVisualization) {
      count++;
      cout << "img num: " << i << endl;
      cv::imshow("window", left_img);
      cv::waitKey(0);
    }
  }
  

  
  return 0;
}

