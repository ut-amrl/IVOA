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
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/LaserScan.h>

#include <vector>
#include <string>
#include <Eigen/Core>
#include "yaml-cpp/yaml.h"

namespace IVOA {
  
// Struct for a 3D point cloud down projected and discretized in a 2D laserscan 
// Compared to a simple 2D scan, this one keeps information about the original
// 3D location of every ray in the 2D scan.
struct ProjectedPtCloud {
  float angle_min;        // start angle of the scan [rad]
  float angle_max;        // end angle of the scan [rad]
  float angle_increment;  // angular distance between measurements [rad]
  float range_min;        // minimum range value [m]
  float range_max;        // maximum range value [m]
  std::vector<float> ranges;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;
};
  
  
class Depth2Pointcloud{
 public:
  Depth2Pointcloud();
  ~Depth2Pointcloud() = default;

  bool GeneratePointcloud(const cv::Mat &depth_img,
                          int img_margin,
                          sensor_msgs::PointCloud2* pointcloud2);
  
  bool GenerateProjectedPtCloud(const cv::Mat &depth_img,
                                int img_margin,      
                                float angle_increment,  
                                float range_min,        
                                float range_max,
                                float height_min,
                                float height_max,
                                ProjectedPtCloud* proj_ptcloud);
  
  sensor_msgs::LaserScan ProjectedPtCloud_to_LaserScan(
                              const ProjectedPtCloud& proj_ptcloud);
  
  sensor_msgs::PointCloud2 FilterPointCloudByDistance(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_distance,
                          const float& max_distance);
  
  sensor_msgs::PointCloud2 FilterPointCloudByHeight(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_height,
                          const float& max_height);
  
  sensor_msgs::PointCloud2 FilterPointCloudByHeightAndDistance(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_height,
                          const float& max_height,
                          const float& min_dist,
                          const float& max_dist);
 
  
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
  geometry_msgs::Point32 Calculate3DCoord(int u, int v, float r);
  
  bool CalculateLaserIndex(float angle_rad,
                           float angle_min,
                           float angle_max,
                           float angle_increment,
                           int *index);

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
