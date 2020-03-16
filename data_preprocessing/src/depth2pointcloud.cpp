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

#include "depth2pointcloud.h"
#include <glog/logging.h>

using std::vector;
using Eigen::Vector3f;
using Eigen::Matrix;
using Eigen::Matrix4f;
using Eigen::Map;
using std::cout;
using std::endl;

namespace IVOA {
Depth2Pointcloud::Depth2Pointcloud() {
  T_cam2base_ = Matrix4f::Identity();
}

bool Depth2Pointcloud::GeneratePointcloud(
    const cv::Mat &depth_img,
    sensor_msgs::PointCloud2* pointcloud2) {
  if (!calibration_is_loaded_) {
    std::cout << "Calibration files are not loaded." << std::endl;
    return false;
  }

  // Pointcloud2
  int max_point_num = depth_img.rows * depth_img.cols;
  pointcloud2->header.frame_id = "base_link";
  pointcloud2->width  = 0;
  pointcloud2->height = 1;
  pointcloud2->is_bigendian = false;
  pointcloud2->is_dense = false; // there may be invalid points
  //for fields setup
  sensor_msgs::PointCloud2Modifier modifier(*pointcloud2);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.reserve(max_point_num);
//   modifier.resize(max_point_num);
  //iterators
  sensor_msgs::PointCloud2Iterator<float> out_x(*pointcloud2, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(*pointcloud2, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(*pointcloud2, "z");

  int valid_points_count = 0;
  for (unsigned int y = 0; y < depth_img.rows; y++) {
    for (unsigned int x = 0; x < depth_img.cols; x++) {      
      float depth = depth_img.at<float>(y, x);

      // Reconstruct 3D point from x, y, raw_depth.
      geometry_msgs::Point32 point;

      point = Calculate3DCoord(x, y, depth);
      Eigen::Vector4f pt_cam;
      pt_cam << point.x, point.y, point.z, 1.0;
//       std::cout << pt_cam.transpose() << std::endl;
      
      // TODO: Here we are assuming that the terrain is flat and that the
      // base_link is parallel to the terrain. The pose of the base_link with
      // respect to the world frame should be taken into accound to be more
      // accurate and consider the robot tilting due to suspension
      
      // Convert the 3d point to the base_link reference frame
      Eigen::Vector4f pt_base = T_cam2base_ * pt_cam;
      point.x = pt_base(0);
      point.y = pt_base(1);
      point.z = pt_base(2);

      if (depth > min_range_ && depth < max_range_) {
        // Pointcloud2
        valid_points_count++;
        pointcloud2->width = valid_points_count;
        modifier.resize(valid_points_count);
        *out_x = point.x;
        *out_y = point.y;
        *out_z = point.z;
        //increment
        ++out_x;
        ++out_y;
        ++out_z;
      }
    }
  }

  return true;
}

cv::Mat Depth2Pointcloud::GenerateObstacleImage(const cv::Mat &depth_img,
                                const float &positive_height_thresh,
                                const float &negative_height_thresh,
                                cv::Mat *obstacle_distance) {
  if (!calibration_is_loaded_) {
    LOG(FATAL) << "Calibration files are not loaded.";
  }

  cv::Mat obstacle_img(depth_img.rows, depth_img.cols, CV_8U);
  cv::Mat obstacle_dist(depth_img.rows, depth_img.cols, CV_32F);
  for (unsigned int y = 0; y < depth_img.rows; y++) {
    for (unsigned int x = 0; x < depth_img.cols; x++) {      
      float depth = depth_img.at<float>(y, x);

      // Reconstruct 3D point from x, y, raw_depth.
      geometry_msgs::Point32 point;

      point = Calculate3DCoord(x, y, depth);
      Eigen::Vector4f pt_cam;
      pt_cam << point.x, point.y, point.z, 1.0;
//       std::cout << pt_cam.transpose() << std::endl;
      
      // TODO: Here we are assuming that the terrain is flat and that the
      // base_link is parallel to the terrain. The pose of the base_link with
      // respect to the world frame should be taken into accound to be more
      // accurate and consider the robot tilting due to suspension
      
      // Convert the 3d point to the base_link reference frame
      Eigen::Vector4f pt_base = T_cam2base_ * pt_cam;
     
       
      if (pt_base(2) > fabs(positive_height_thresh) ||
          pt_base(2) < -fabs(negative_height_thresh)) {
        obstacle_img.at<uint8_t>(y, x) = 1;
        obstacle_dist.at<float>(y,x) = sqrt(pt_base(0) * pt_base(0) 
                                          + pt_base(1) * pt_base(1));
      } else {
        obstacle_img.at<uint8_t>(y, x) = 0;
        obstacle_dist.at<float>(y,x) = -1;
      }
    }
  }

  *obstacle_distance = obstacle_dist.clone();
  return obstacle_img;
}

int Depth2Pointcloud::LoadCameraCalibration(
  const std::string extrinsics_file) {
  // Load the calibration yaml files
  YAML::Node cam_ext = YAML::LoadFile(extrinsics_file);


  std::vector<float> T_cam2base_vec;
  if (cam_ext["T_cam2base"]["data"]) {
    T_cam2base_vec = cam_ext["T_cam2base"]["data"].as<std::vector<float>>();
  } else {
    std::cout << "Cannot read " << extrinsics_file << std::endl;
    return -1;
  }


  CHECK_EQ(T_cam2base_vec.size(), 12) << "Corrupted calibration file.";


  // Convert the loaded data to eigen matrices
  Map<Matrix<float, 3, 4, Eigen::RowMajor>> T_cam2base(T_cam2base_vec.data());
 
  T_cam2base_.topLeftCorner(3, 4) = T_cam2base;
 

  calibration_is_loaded_ = true;
  return 0;
}


geometry_msgs::Point32 Depth2Pointcloud::Calculate3DCoord(
                                                        unsigned int u,
                                                        unsigned int v,
                                                        float raw_depth) {
  // Reconstruct 3D point from x, y, raw_depth.
  geometry_msgs::Point32 point;

  double depth = static_cast<double>(raw_depth);

  // Calculate the 3D coordinates of the point in the optical reference frame
  // of the camera (x points to the right of the image, y to the bottom and z
  // points into the image plane)
  point.x = (u - px_) * depth / fx_;
  point.y = (v - py_) * depth / fy_;
  point.z = depth;

  return point;
}




} // namespace IVOA
