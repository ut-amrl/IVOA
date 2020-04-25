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

#include "evaluator.h"

using std::vector;
using Eigen::Vector3f;
using Eigen::Vector2f;
using Eigen::Matrix;
using Eigen::Matrix4f;
using Eigen::Map;
using std::cout;
using std::endl;

namespace IVOA {
  
Evaluator::Evaluator(float distance_err_thresh,
                     float rel_distance_err_thresh,
                     bool debug_mode):
        distance_err_thresh_ (distance_err_thresh),
        rel_distance_err_thresh_(rel_distance_err_thresh),
        debug_mode_(debug_mode){
          
  cam_mat_ << fx_,  0.0,  px_,
              0.0,  fy_,  py_,
              0.0,  0.0,  1.0;
  
  prediction_label_counts_.resize(4);
  for (int i = 0; i < 4; i++) {
    prediction_label_counts_[i] = 0;
  }
}

int Evaluator::LoadCameraCalibration(
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

unsigned int Evaluator::EvaluatePredictions(const ProjectedPtCloud& pred_scan,
                                    const ProjectedPtCloud& gt_scan,
                                    const Eigen::Matrix4f& T_base2map,
                                    const unsigned long int& frame_id) {
  CHECK_EQ(pred_scan.ranges.size(), gt_scan.ranges.size());
  
  // Populate the error laser scans in debug mode
  if (debug_mode_) {
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "base_link";
    fp_scan_.header = fn_scan_.header = header;
    fp_scan_.angle_min = fn_scan_.angle_min = gt_scan.angle_min;
    fp_scan_.angle_max = fn_scan_.angle_max = gt_scan.angle_max;
    fp_scan_.range_max = fn_scan_.range_max = gt_scan.range_max;
    fp_scan_.range_min = fn_scan_.range_min = gt_scan.range_min;
    fp_scan_.angle_increment = fn_scan_.angle_increment = 
                               gt_scan.angle_increment;
    
    fp_scan_.time_increment = fn_scan_.time_increment = 0;
    fp_scan_.scan_time = fn_scan_.scan_time = 0;
    vector<float> init_ranges(gt_scan.ranges.size(), 2 * gt_scan.range_max);
    fp_scan_.ranges = fn_scan_.ranges = init_ranges;
  }
  
  std::vector<Error> errors;
  
  for(size_t i = 0; i < gt_scan.ranges.size(); i++) {
    if (gt_scan.ranges[i] < gt_scan.range_min) {
      continue;
    }
   
    // Both prediction and gt indicate out of range: TN
    if (gt_scan.ranges[i] > gt_scan.range_max &&
        pred_scan.ranges[i] > gt_scan.range_max) {
      prediction_label_counts_[TN]++;
      continue;
    }    
   
    bool error_found = false;
    PredictionLabel error_type;
    float dist_err = gt_scan.ranges[i] - pred_scan.ranges[i];
    float rel_err_thresh = rel_distance_err_thresh_ * gt_scan.ranges[i];
   
    if (dist_err > std::max(distance_err_thresh_, rel_err_thresh)) {
      // FP
      prediction_label_counts_[FP]++;
      error_found = true;
      error_type = FP;
      if (debug_mode_) {
        fp_scan_.ranges[i] = pred_scan.ranges[i];
      }
    } else if (dist_err < -std::max(distance_err_thresh_, rel_err_thresh)) {
      // FN
      prediction_label_counts_[FN]++;
      error_found = true;
      error_type = FN;
      if (debug_mode_) {
        fn_scan_.ranges[i] = gt_scan.ranges[i];
      }
    } else {
      // TP
      prediction_label_counts_[TP]++;
    }
    
    if (error_found) {
      Vector3f error_loc_in_map;
      Vector2f error_pixel_coord;
      
      LocateError(pred_scan.points[i],
                   gt_scan.points[i],
                   error_type,
                   T_base2map,
                   &error_loc_in_map,
                   &error_pixel_coord);
      
      errors.push_back(Error(error_type, 
                             error_loc_in_map,
                             error_pixel_coord,
                             frame_id));
    }
  }
  
  errors_list_.push_back(errors);
  frame_count++;
  return errors_list_.size() - 1;
}

std::vector<unsigned long int> Evaluator::GetStatistics() {
  return prediction_label_counts_;
}

std::vector<std::vector<Error>> Evaluator::GetErrors() {
  return errors_list_;
}

sensor_msgs::LaserScan Evaluator::GetFalsePositivesScan() {
  if (!debug_mode_) {
    LOG(WARNING) << "Error laserscans are only generated in debug mode.";
  }
  return fp_scan_;
}

sensor_msgs::LaserScan Evaluator::GetFalseNegativesScan() {
  if (!debug_mode_) {
    LOG(WARNING) << "Error laserscans are only generated in debug mode.";
  }
  return fn_scan_;
}


void Evaluator::LocateError(const Eigen::Vector3f& pred_loc_in_cam,
                   const Eigen::Vector3f& gt_loc_in_cam,
                   PredictionLabel error_type,
                   const Eigen::Matrix4f& T_base2map,
                   Eigen::Vector3f* err_loc_map,
                   Eigen::Vector2f* err_pixel_coord) {
  Vector3f pt_of_interest;
  if (error_type == FP) {
    pt_of_interest = pred_loc_in_cam;
  } else if (error_type == FN) {
    pt_of_interest = gt_loc_in_cam;
  } else {
    LOG(FATAL) << "Unknown error type!";
  }
  
  *err_pixel_coord = ProjectToCam(pt_of_interest);
  
  if (!calibration_is_loaded_) {
    LOG(FATAL) << "Camera calibration was not provided!";
  }
  
  Eigen::Vector4f pt_of_interest_h(1.0, 1.0, 1.0, 1.0);
  pt_of_interest_h.head(3) = pt_of_interest;
  Eigen::Vector4f pt_in_map;
  pt_in_map = T_base2map * T_cam2base_ * pt_of_interest_h;
  *err_loc_map = pt_in_map.head(3);
}

Eigen::Vector2f Evaluator::ProjectToCam(
                          const Eigen::Vector3f& point_3d_in_cam_ref) {
  CHECK_EQ(point_3d_in_cam_ref.size(), 3);

  if (point_3d_in_cam_ref(2) == 0) {
    LOG(WARNING) << "The provided map point has a depth of 0!" ;
    return Vector2f(0, 0);
  }
  Vector3f projection = (cam_mat_ * point_3d_in_cam_ref)/point_3d_in_cam_ref(2);
  return projection.head(2);
}

} // namespace IVOA


