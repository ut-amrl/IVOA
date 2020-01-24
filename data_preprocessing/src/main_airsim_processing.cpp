// Copyright 2020 srabiee@cs.utexas.edu
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
#include "dataset.h"
#include "depth2pointcloud.h"
#include"cnpy.h"

using cv::Mat;
using namespace std;
using namespace IVOA;
using std::vector;
using cv::Point;

// Parameters
const float kPositiveHeightObsThresh = 0.3; // meters
const float kNegativeHeightObsThresh = 0.3; // meters
const bool kVisualization = false;
const float kPatchSize = 100;
const float kPatchStride = 30;
const double kObstacleRatioThresh = 0.05;
const cv::Size kImageSize(960, 600);


vector<Point> GenerateQueryPoints(const cv::Size img_size,
                                  const float &patch_size,
                                  const float &stride) {
  vector<Point> query_points;
  float x = 0;
  float y = 0;
 
  
  for (x = patch_size / 2 + 1; x + patch_size/2 < img_size.width; x+=stride) {
    for (y = patch_size / 2 + 1; 
         y + patch_size/2 < img_size.height; 
         y+=stride) {
      query_points.push_back(Point(x, y));
    }
  }
  
  return query_points;
}


int main(int argc, char **argv) {
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;   // INFO level logging.
  FLAGS_colorlogtostderr = 1;  // Colored logging.
  FLAGS_logtostderr = true;    // Don't log to disk
  
  ros::init(argc, argv, "IVOA_data_preprocessing");
  ros::NodeHandle nh;
  
  //TODO: Read command line arguments
  string base_dir = "/media/ssd2/datasets/AirSim_IVOA/initial_ml_tool";
  int session_num = 0;
  string cam_extrinsics_path = 
              "/home/srabiee/My_Repos/IVOA/data_preprocessing/"
              "util/Camera_Extrinsics.yaml";
  string output_dataset_dir = "/media/ssd2/datasets/AirSim_IVOA/airsim_ivoa/";

  
  
  
  ros::Publisher point_cloud_publisher_gt =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/gt_pointcloud", 1);
  ros::Publisher point_cloud_publisher_pred =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/pred_pointcloud", 1);
  
  
  // Using only one instance of Depth2Pointcloud since the depth and left RGB
  // camera share the same extrinsics and intrinsics in our AirSim setup 
  Depth2Pointcloud depth_img_converter;
  depth_img_converter.LoadCameraCalibration(cam_extrinsics_path);
  
  //TODO: Read img_num from the img_depth directory
  int img_num = 50;
  string gt_depth_dir = base_dir + "/img_depth/";
  string pred_depth_dir = base_dir + "/img_left/";
  string left_img_dir = base_dir + "/img_left/";
 
  
  Dataset dataset(kPatchSize,
                  session_num,
                  output_dataset_dir,
                  kObstacleRatioThresh);

  vector<Point> query_points = GenerateQueryPoints(kImageSize,
                                                   kPatchSize,
                                                   kPatchStride);

  dataset.LoadQueryPoints(query_points);
  
  for (int i = 0; i < img_num; i++) {
    cout << i << endl;
    
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
    cv::Mat depth_img_pred = cv::Mat(arr.shape[2],
                                     arr.shape[3],
                                     CV_32F,
                                     loaded_data);
  
    
    // Resize and linearly interpolate the predicted depth image so that it 
    // is the same size as the ground truth depth image
    cv::resize(depth_img_pred,depth_img_pred,depth_img_gt.size());
    

    cv::Mat obstacle_img_gt = depth_img_converter.GenerateObstacleImage(
                                              depth_img_gt,
                                              kPositiveHeightObsThresh,
                                              kNegativeHeightObsThresh);
    cv::Mat obstacle_img_pred = depth_img_converter.GenerateObstacleImage(
                                              depth_img_pred,
                                              kPositiveHeightObsThresh,
                                              kNegativeHeightObsThresh);

    dataset.LabelData(obstacle_img_gt, obstacle_img_pred);
    dataset.SaveImages(left_img);
    
    if (kVisualization) {
      sensor_msgs::PointCloud2 pointcloud2;
      if (depth_img_converter.GeneratePointcloud(depth_img_gt,
                                                &pointcloud2)) {
        point_cloud_publisher_gt.publish(pointcloud2);
      }
      
      if (depth_img_converter.GeneratePointcloud(depth_img_pred,
                                                &pointcloud2)) {
        point_cloud_publisher_pred.publish(pointcloud2);
      }
    }

    
//     cout << "img num: " << i << endl;
//       // Visualize and verify the depth image
//     double min;
//     double max;
//     cv::minMaxIdx(depth_img_gt, &min, &max);
//     cv::Mat adjMap;
//     cv::convertScaleAbs(depth_img_gt, adjMap, 255.0 / max);
//     cv::imshow("window", adjMap); 
//     cv::waitKey(0);

  }
  

  dataset.SaveToFile();
  
  return 0;
}
