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

DEFINE_double(ground_plane_height, 0.6, "Height of the ground plane "
              "in meters. This can be environment dependent.");

const int EPSILON = 1e-5;

namespace IVOA {
Depth2Pointcloud::Depth2Pointcloud() {
  T_cam2base_ = Matrix4f::Identity();
}

bool Depth2Pointcloud::GeneratePointcloud(
    const cv::Mat &depth_img,
    float img_margin_perc,
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
  float img_margin = img_margin_perc * depth_img.cols / 100.0f;
  for (int y = img_margin; y < depth_img.rows - img_margin; y++) {
    for (int x = img_margin; x < depth_img.cols - img_margin; x++) {      
      float depth = depth_img.at<float>(y, x);

      // Reconstruct 3D point from x, y, raw_depth.
      geometry_msgs::Point32 point;

      point = Calculate3DCoord(x, y, depth);
      Eigen::Vector4f pt_cam;
      pt_cam << point.x, point.y, point.z, 1.0;
//       std::cout << pt_cam.transpose() << std::endl;
     
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


bool Depth2Pointcloud::GenerateProjectedPtCloud(const cv::Mat &depth_img,
                                            float img_margin_perc,      
                                            float angle_increment,  
                                            float range_min,        
                                            float range_max,
                                            float height_min,
                                            float height_max,
                                            const Eigen::Matrix4f& T_base2map,
                                            ProjectedPtCloud* proj_ptcloud) {
  if (!calibration_is_loaded_) {
    std::cout << "Calibration files are not loaded." << std::endl;
    return false;
  }
  // TODO: The ground plane height might be environment specific. Detecting
  // the obstacles given the terrain roughness, the point cloud in the 
  // base_link in conjuction with taking into accout the camera orientation
  // seems to be the most general.
  
  float img_margin = img_margin_perc * depth_img.cols / 100.0f;
  float angle_max = atan((depth_img.cols - img_margin - cam_mat_(0,2)) / 
                                                        cam_mat_(0,0));
  float angle_min = -angle_max;
  
  int laserscan_size = floor((angle_max - angle_min) / angle_increment);
 
  proj_ptcloud->ranges.clear();
  proj_ptcloud->points.clear();
  proj_ptcloud->ranges.resize(laserscan_size);
  proj_ptcloud->points.resize(laserscan_size);
  proj_ptcloud->angle_max = angle_max;
  proj_ptcloud->angle_min = angle_min;
  proj_ptcloud->angle_increment = angle_increment;
  proj_ptcloud->range_max = range_max;
  proj_ptcloud->range_min = range_min;
  for (int i = 0; i < laserscan_size; i++) {
    proj_ptcloud->ranges[i] = range_max + 0.01f;
    
  }
  
  for (int y = img_margin; y < depth_img.rows - img_margin; y++) {
    for (int x = img_margin; x < depth_img.cols - img_margin; x++) {      
      float depth = depth_img.at<float>(y, x);

      // Reconstruct 3D point from x, y, raw_depth.
      geometry_msgs::Point32 point;

      point = Calculate3DCoord(x, y, depth);
      Eigen::Vector4f pt_cam;
      pt_cam << point.x, point.y, point.z, 1.0;
           
      // Convert the 3d point to the base_link reference frame
      Eigen::Vector4f pt_base = T_cam2base_ * pt_cam;
    
      // Convert the 3d point to the map reference frame
      Eigen::Vector4f pt_map = T_base2map * pt_base;
      
      
                
      // TODO: Currently obstacles are detected given their height in the 
      //  map reference frame. This is only accurate for flat maps. For
      //  a more general solution, you can consider the terrain roughness.
      
      // TODO: For points with z < -min_height, project them to the ground
      // plane along the line to camera center. (This is to handle hole
      // locations more accurately)
      if (fabs(pt_map(2) - FLAGS_ground_plane_height) >= height_min 
          && (pt_map(2) - FLAGS_ground_plane_height) <= height_max) {
        float angle = atan2(pt_base(1), pt_base(0));
        int index = 0;
        if(!CalculateLaserIndex(angle,
                                angle_min,
                                angle_max,
                                angle_increment,
                                &index)) {
          // The point is out of range of the laserscan
          continue;
        }

        float range = sqrt(pt_base(0) * pt_base(0) 
                          + pt_base(1) * pt_base(1));
        if (range < proj_ptcloud->ranges[index] 
            && range >= range_min
            && range <= range_max) {
          proj_ptcloud->ranges[index] = range;
          proj_ptcloud->points[index] = Eigen::Vector3f(pt_cam(0),
                                                        pt_cam(1),
                                                        pt_cam(2));
        }
      }
    }
  }

  return true;
}


sensor_msgs::LaserScan Depth2Pointcloud::ProjectedPtCloud_to_LaserScan(
                              const ProjectedPtCloud& proj_ptcloud) {
  sensor_msgs::LaserScan laserscan;
  laserscan.header.stamp = ros::Time::now();
  laserscan.header.frame_id = "base_link";
  
  laserscan.angle_min = proj_ptcloud.angle_min;
  laserscan.angle_max = proj_ptcloud.angle_max;
  laserscan.range_max = proj_ptcloud.range_max;
  laserscan.range_min = proj_ptcloud.range_min;
  laserscan.angle_increment = proj_ptcloud.angle_increment;
  
  laserscan.time_increment = 0;
  laserscan.scan_time = 0;
  laserscan.ranges = proj_ptcloud.ranges;
  
  return laserscan;
}


sensor_msgs::PointCloud2 Depth2Pointcloud::FilterPointCloudByDistance(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_distance,
                          const float& max_distance){
  sensor_msgs::PointCloud2 filtered_ptcloud = pointcloud2;
  sensor_msgs::PointCloud2Iterator<float> out_x(filtered_ptcloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(filtered_ptcloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(filtered_ptcloud, "z");

  for (sensor_msgs::PointCloud2Iterator<float> it(filtered_ptcloud, "x"); 
       it != it.end(); 
       ++it) {
    float x = it[0];
    float y = it[1];
    float distance = sqrt(x * x + y * y);
    if (distance < min_distance || distance > max_distance) {
      it[0] = std::numeric_limits<float>::quiet_NaN();
      it[1] = std::numeric_limits<float>::quiet_NaN();
      it[2] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  
  return filtered_ptcloud;
}
  
sensor_msgs::PointCloud2 Depth2Pointcloud::FilterPointCloudByHeight(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_height,
                          const float& max_height){
  sensor_msgs::PointCloud2 filtered_ptcloud = pointcloud2;
  sensor_msgs::PointCloud2Iterator<float> out_x(filtered_ptcloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(filtered_ptcloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(filtered_ptcloud, "z");

  for (sensor_msgs::PointCloud2Iterator<float> it(filtered_ptcloud, "x"); 
       it != it.end(); 
       ++it) {
    float z = it[2];
    if (z < min_height || z > max_height) {
      it[0] = std::numeric_limits<float>::quiet_NaN();
      it[1] = std::numeric_limits<float>::quiet_NaN();
      it[2] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  return filtered_ptcloud;
}

sensor_msgs::PointCloud2 Depth2Pointcloud::FilterPointCloudByHeightAndDistance(
                          const sensor_msgs::PointCloud2& pointcloud2,
                          const float& min_height,
                          const float& max_height,
                          const float& min_dist,
                          const float& max_dist){
  sensor_msgs::PointCloud2 filtered_ptcloud = pointcloud2;
  sensor_msgs::PointCloud2Iterator<float> out_x(filtered_ptcloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(filtered_ptcloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(filtered_ptcloud, "z");

  for (sensor_msgs::PointCloud2Iterator<float> it(filtered_ptcloud, "x"); 
       it != it.end(); 
       ++it) {
    float x = it[0];
    float y = it[1];
    float z = it[2];
    float distance = sqrt(x * x + y * y);
    if (z < min_height || z > max_height ||
        distance < min_dist || distance > max_dist) {
      it[0] = std::numeric_limits<float>::quiet_NaN();
      it[1] = std::numeric_limits<float>::quiet_NaN();
      it[2] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  return filtered_ptcloud;
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
  for (int y = 0; y < depth_img.rows; y++) {
    for (int x = 0; x < depth_img.cols; x++) {      
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
  const std::string calibration_file) {
  // Load the calibration yaml files
  YAML::Node cam_cal = YAML::LoadFile(calibration_file);


  std::vector<float> T_cam2base_vec;
  if (cam_cal["T_cam2base"]["data"]) {
    T_cam2base_vec = cam_cal["T_cam2base"]["data"].as<std::vector<float>>();
  } else {
    std::cout << "Cannot read " << calibration_file << std::endl;
    return -1;
  }
  CHECK_EQ(T_cam2base_vec.size(), 12) << "Corrupted calibration file.";
  // Convert the loaded data to eigen matrices
  Map<Matrix<float, 3, 4, Eigen::RowMajor>> T_cam2base(T_cam2base_vec.data());

  T_cam2base_.topLeftCorner(3, 4) = T_cam2base;
 
  std::vector<float> cam_mat;
  if (cam_cal["DEPTH.K"]["data"]) {
    cam_mat = cam_cal["DEPTH.K"]["data"].as<std::vector<float>>();
  } else {
    std::cout << "Cannot read " << calibration_file << std::endl;
    return -1;
  }
  
  CHECK_EQ(cam_mat.size(), 9) << "Corrupted calibration file.";
  Map<Matrix<float, 3, 3, Eigen::RowMajor>> cam_mat_eig(cam_mat.data());
  cam_mat_ = cam_mat_eig;
 
  
  calibration_is_loaded_ = true;
  return 0;
}


geometry_msgs::Point32 Depth2Pointcloud::Calculate3DCoord(
                                                        int u,
                                                        int v,
                                                        float raw_depth) {
  // Reconstruct 3D point from x, y, raw_depth.
  geometry_msgs::Point32 point;

  double depth = static_cast<double>(raw_depth);

  // Calculate the 3D coordinates of the point in the optical reference frame
  // of the camera (x points to the right of the image, y to the bottom and z
  // points into the image plane)
  point.x = (u - cam_mat_(0,2)) * depth / cam_mat_(0,0);
  point.y = (v - cam_mat_(1,2)) * depth / cam_mat_(1,1);
  point.z = depth;

  return point;
}

bool Depth2Pointcloud::CalculateLaserIndex(float angle_rad,
                                          float angle_min,
                                          float angle_max,
                                          float angle_increment,
                                          int *index) {
  if ((angle_rad > (angle_max + angle_increment / 2)) ||
      (angle_rad < (angle_min - angle_increment / 2))) {
    return false;
  }

  *index = floor ((angle_rad - (angle_min - angle_increment /2))
                  / angle_increment);
  return true;
}




} // namespace IVOA
