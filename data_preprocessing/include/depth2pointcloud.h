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
#ifndef IVOA_DEPTH2POINTCLOUD_
#define IVOA_DEPTH2POINTCLOUD_

#include <ros/ros.h>
#include <ros/console.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <vector>
#include <string>
#include <Eigen/Core>
#include "yaml-cpp/yaml.h"

namespace IVOA {
class Depth2Pointcloud{
 public:
  Depth2Pointcloud();
  ~Depth2Pointcloud() = default;

  bool GeneratePointcloud(const cv::Mat &depth_img,
                          sensor_msgs::PointCloud2* pointcloud2);
  
  sensor_msgs::PointCloud2 FilterPointCloudByDistance(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_distance,
                          const float& max_distance);
  
  sensor_msgs::PointCloud2 FilterPointCloudByHeight(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_height,
                          const float& max_height);
 
  
  // Given the depth image, the extrinsic calibration and the obstacle height
  // thresholds, it generates a binary image where each pixel is calssified as
  // obstacle (1) or traversable (0)
  cv::Mat GenerateObstacleImage(const cv::Mat &depth_img,
                                const float &positive_height_thresh,
                                const float &negative_height_thresh,
                                cv::Mat *obstacle_distance);

  int LoadCameraCalibration(const std::string extrinsics_file);


 private:
  // Given the pixel coordinates and the depth reading, it generates the
  // corresponding 3d point (in meters)
  geometry_msgs::Point32 Calculate3DCoord(unsigned int u,
                                          unsigned int v,
                                          float r);

  // Camera intrinsics 
  float fx_ = 480.0;
  float fy_ = 480.0;
  float px_ = 480.0;
  float py_ = 300.0;

  float min_range_ = 0.0; // m
  float max_range_ = 1e10; // m


  // Transformation from camera frame to robot's base frame
  Eigen::Matrix4f T_cam2base_;
  bool calibration_is_loaded_ = false;


};
} // namespace IVOA

#endif // IVOA_DEPTH2POINTCLOUD_
