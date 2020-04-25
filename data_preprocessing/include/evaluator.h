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
#ifndef IVOA_EVALUATOR_
#define IVOA_EVALUATOR_

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/LaserScan.h>

#include <vector>
#include <string>
#include <Eigen/Core>
#include <glog/logging.h>

#include "depth2pointcloud.h"

namespace IVOA {
  

class Evaluator{
 public:
  Evaluator(float distance_err_thresh,
            float rel_distance_err_thresh,
            bool debug_mode);
  ~Evaluator() = default;

  enum PredictionLabel {
    TP = 0,
    TN = 1,
    FP = 2,
    FN = 3
  };
  
  struct Error {
    PredictionLabel error_type;
    Eigen::Vector3f loc_map;
    Eigen::Vector2f pixel_coord;
    unsigned long int frame_id;
    
    Error(PredictionLabel error_type, 
          Eigen::Vector3f loc_map,
          Eigen::Vector2f pixel_coord,
          unsigned long int frame_id):
          error_type(error_type),
          loc_map(loc_map),
          pixel_coord(pixel_coord),
          frame_id(frame_id){}
  };

  struct ErrorTrack {
    PredictionLabel error_type;
    std::vector<std::pair<unsigned long int, Eigen::Vector3f>> loc_map_history; // pairs of (frame_id, location)
    unsigned long int last_frame_id; // to keep track of the last time we saw this error
  };

  int LoadCameraCalibration(const std::string extrinsics_file);
  // Returns the index in the errors_list_ of this evaluation
  unsigned int EvaluatePredictions(const ProjectedPtCloud& pred_scan,
                           const ProjectedPtCloud& gt_scan,
                           const Eigen::Matrix4f& T_base2map,
                           const unsigned long int& frame_id);
 
  std::vector<unsigned long int> GetStatistics();
  
  // TODO: Add helper functions for visualization of the errors 
  sensor_msgs::LaserScan GetFalsePositivesScan();
  sensor_msgs::LaserScan GetFalseNegativesScan();

  std::vector<std::vector<Error>> GetErrors();
  std::vector<ErrorTrack> GetErrorTracks();
  
 private:
  
  // Given the 3D location of a pair of points from the predicted depth and 
  // ground truth depth (both in the camera coordinate frame), it calculates 
  // the location of the error in the map
  // coordinate frame as well as the corresponding pixel coordinate in 
  // current frame
  void LocateError(const Eigen::Vector3f& pred_loc_in_cam,
                   const Eigen::Vector3f& gt_loc_in_cam,
                   PredictionLabel error_type,
                   const Eigen::Matrix4f& T_base2map,
                   Eigen::Vector3f* err_loc_map,
                   Eigen::Vector2f* err_pixel_coord);
  
  // Projects a 3D point that is in the reference frame of the camera to the
  // image plane
  Eigen::Vector2f ProjectToCam(const Eigen::Vector3f& point_3d_in_cam_ref);
  
  std::vector<unsigned long int>prediction_label_counts_;
  unsigned long int frame_count = 0;
  std::vector<std::vector<Error>> errors_list_;
  std::vector<ErrorTrack> error_tracks_;
  
  // Laserscans for visualizing FP and FNs. Only used in debug_mode
  sensor_msgs::LaserScan fp_scan_;
  sensor_msgs::LaserScan fn_scan_;

// If the predicted distance to obstacle is off from the ground truth by
// more than max(distance_err_thresh_, rel_distance_err_thresh_ *TrueDistance), 
// it will be labeled as either FP (if pred_dist < gt_dist) or 
// FN (if pred_dist > gt_dist)
  float distance_err_thresh_;
  float rel_distance_err_thresh_;
  
  // TODO: Load camera intrinsics from file
  // Camera intrinsics 
  float fx_ = 480.0;
  float fy_ = 480.0;
  float px_ = 480.0;
  float py_ = 300.0;
  
  bool debug_mode_;
  
  // Camera matrix 
  Eigen::Matrix3f cam_mat_;
  
  // Transformation from camera frame to robot's base frame
  Eigen::Matrix4f T_cam2base_;
  bool calibration_is_loaded_ = false;

  static const unsigned int MAX_ERROR_TRACK_GAP=5;
  static constexpr float MAX_ERROR_TRACK_MAP_DISTANCE=3.0f;
};
} // namespace IVOA

#endif // IVOA_EVALUATOR_

