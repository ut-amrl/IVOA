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
#include <numeric>

using std::vector;
using Eigen::Vector3f;
using Eigen::Vector2f;
using Eigen::Matrix;
using Eigen::Matrix4f;
using Eigen::Map;
using std::cout;
using std::endl;

#define DEBUG false

namespace IVOA {

const std::vector<float> Evaluator::PCT_WINDOWS = {0.3f, 0.5f, 0.7f};

Evaluator::Evaluator(float distance_err_thresh,
                     float rel_distance_err_thresh,
                     bool debug_mode):
        distance_err_thresh_ (distance_err_thresh),
        rel_distance_err_thresh_(rel_distance_err_thresh),
        debug_mode_(debug_mode){
  
  prediction_label_counts_.resize(4);
  for (int i = 0; i < 4; i++) {
    prediction_label_counts_[i] = 0;
  }
}

int Evaluator::LoadCameraCalibration(
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
  T_cam2base_ = Matrix4f::Identity();
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

unsigned int Evaluator::EvaluatePredictions(const ProjectedPtCloud& pred_scan,
                                    const ProjectedPtCloud& gt_scan,
                                    const Eigen::Matrix4f& T_base2map,
                                    const unsigned long int& frame_id,
                                    const cv::Mat &depth_img_gt) {
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
    vector<float> init_ranges(gt_scan.ranges.size(), gt_scan.range_max + 0.01f);
    fp_scan_.ranges = fn_scan_.ranges = init_ranges;
  }
  
  std::vector<Error> errors;
  
  latest_frame_dist_errors_.clear();
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
    float rel_dist_err = dist_err / gt_scan.ranges[i];
    float rel_err_thresh = rel_distance_err_thresh_ * gt_scan.ranges[i];
    latest_frame_dist_errors_.push_back(-dist_err);
    dist_errors_.push_back(-dist_err); // for visualization we want prediction-actual
    rel_dist_errors_.push_back(-rel_dist_err);

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
                   depth_img_gt,
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


  // Error Tracking
  std::vector<int> tmp(errors.size());
  size_t n = 0;
  std::generate(tmp.begin(),tmp.end(),[n]()mutable{return n++;});
  std::set<int> untracked_errors(tmp.begin(), tmp.end());

  #if DEBUG
  printf("Looping over %d existing tracks for %d errors\n", error_tracks_.size(), errors.size());
  #endif
  
  for(ErrorTrack& track : error_tracks_) {
    // printf("IN HERE FRAMES %ld %ld\t", track.last_frame_id, frame_id);
    if ((frame_id - track.last_frame_id) > Evaluator::MAX_ERROR_TRACK_GAP) {
      continue;
    }

    // Anchor each error track to its first seen location
    // alternatives include centroid, or most recent location
    Eigen::Vector3f track_loc = track.loc_map_history[0].second;

    // Find the closest untracked error of the appropriate class
    float min_distance = Evaluator::MAX_ERROR_TRACK_MAP_DISTANCE;
    int min_idx = -1;
    for(int i : untracked_errors) {
      if (track.error_type != errors[i].error_type) {
        continue;
      }

      float distance = (errors[i].loc_map - track_loc).norm();
      if (distance < min_distance) {
        min_distance = distance;
        min_idx = i;
      }
    }
    // printf("found min %d %f\n", min_idx, min_distance);

    // Add the found error to the track
    if (min_idx != -1) {
      untracked_errors.erase(min_idx);
      track.last_frame_id = frame_id;
      track.loc_map_history.push_back(std::make_pair(frame_id, errors[min_idx].loc_map));
    }
  }

  #if DEBUG
  printf("Creating tracks for %d remaining errors\n", untracked_errors.size());
  #endif

  // For any errors *still* untracked, create new tracks!
  for(int i : untracked_errors) {
    ErrorTrack track;
    track.error_type = errors[i].error_type;
    track.last_frame_id = frame_id;
    track.loc_map_history = std::vector<std::pair<unsigned long int, Eigen::Vector3f>>();
    track.loc_map_history.push_back(std::make_pair(frame_id, errors[i].loc_map));
    error_tracks_.push_back(track);
  }

  #if DEBUG
  printf("Finished error track handling\n");
  #endif

  return errors_list_.size() - 1;
}

std::vector<unsigned long int> Evaluator::GetStatistics() {
  return prediction_label_counts_;
}

std::vector<std::vector<Evaluator::Error>>& Evaluator::GetErrors() {
  return errors_list_;
}

std::vector<Evaluator:: ErrorTrack>& Evaluator::GetErrorTracks() {
  return error_tracks_;
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

sensor_msgs::PointCloud2 Evaluator::GetErrorsPointCloud() {
  sensor_msgs::PointCloud2 ptcloud;
  int ptcloud_size = static_cast<int>(errors_list_.back().size());
  ptcloud.header.frame_id = "map";
  ptcloud.width  = ptcloud_size;
  ptcloud.height = 1;
  ptcloud.is_bigendian = false;
  ptcloud.is_dense = false; // there may be invalid points
    sensor_msgs::PointCloud2Modifier modifier(ptcloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.reserve(ptcloud_size);
  
  sensor_msgs::PointCloud2Iterator<float> out_x(ptcloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(ptcloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(ptcloud, "z");

  for (const Error& err : errors_list_.back()) { 
    *out_x = err.loc_map.x();
    *out_y = err.loc_map.y();
    *out_z = err.loc_map.z();
    ++out_x;
    ++out_y;
    ++out_z;
  }
  
  return ptcloud;
}

float Evaluator::GetLatestFrameMedianAbsError(){
  size_t obs_num = latest_frame_dist_errors_.size();
  
  vector<float> abs_err_values(obs_num);
  for (size_t i = 0; i < obs_num; i++) {
    abs_err_values[i] = fabs(latest_frame_dist_errors_[i]);
  }
  
  if (obs_num == 0) {
    return 0.0;
  } else {
    sort(abs_err_values.begin(), abs_err_values.end());
    if (obs_num % 2 == 0) {
      return (abs_err_values[obs_num / 2 - 1] 
              + abs_err_values[obs_num / 2]) / 2;
    }
    else {
      return abs_err_values[obs_num / 2];
    }
  }
}

float Evaluator::GetLatestFrameMeanAbsError(){
  size_t obs_num = latest_frame_dist_errors_.size();
  
  vector<float> abs_err_values(obs_num);
  for (size_t i = 0; i < obs_num; i++) {
    abs_err_values[i] = fabs(latest_frame_dist_errors_[i]);
  }
  
  if (obs_num == 0) {
    return 0.0;
  } else {
    float sum_abs_err = 0.0;
    for (size_t j = 0; j < abs_err_values.size(); j++) {
      sum_abs_err += abs_err_values[j];
    }
    return sum_abs_err / static_cast<float>(obs_num);
  }
}

void Evaluator::LocateError(const Eigen::Vector3f& pred_loc_in_cam,
                   const Eigen::Vector3f& gt_loc_in_cam,
                   PredictionLabel error_type,
                   const Eigen::Matrix4f& T_base2map,
                   const cv::Mat &depth_img_gt,
                   Eigen::Vector3f* err_loc_map,
                   Eigen::Vector2f* err_pixel_coord) {
  if (!calibration_is_loaded_) {
    LOG(FATAL) << "Camera calibration was not provided!";
  }
  
  Vector3f pt_of_interest;
  if (error_type == FP) {
    *err_pixel_coord = ProjectToCam(pred_loc_in_cam); 
    int u = static_cast<int>(err_pixel_coord->x());
    int v = static_cast<int>(err_pixel_coord->y());
    pt_of_interest = Calculate3DCoord(u, v, depth_img_gt.at<float>(v, u));
  } else if (error_type == FN) {
    pt_of_interest = gt_loc_in_cam;
    *err_pixel_coord = ProjectToCam(gt_loc_in_cam);
  } else {
    LOG(FATAL) << "Unknown error type!";
  }
 
  
  Eigen::Vector4f pt_of_interest_h(1.0, 1.0, 1.0, 1.0);
  pt_of_interest_h.head(3) = pt_of_interest;
  Eigen::Vector4f pt_in_map;
  // std::cout << "point of interest: " << pt_of_interest_h.transpose() << std::endl;
  pt_in_map = T_base2map * T_cam2base_ * pt_of_interest_h;
  // std::cout << "point in map: " << pt_in_map.transpose() << std::endl;
  *err_loc_map = pt_in_map.head(3) / pt_in_map[3];
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

Eigen::Vector3f Evaluator::Calculate3DCoord(int u,
                                            int v,
                                            float raw_depth) {
  // Reconstruct 3D point from x, y, raw_depth.
  Vector3f point;

  double depth = static_cast<double>(raw_depth);

  // Calculate the 3D coordinates of the point in the optical reference frame
  // of the camera (x points to the right of the image, y to the bottom and z
  // points into the image plane)
  point.x() = (u - cam_mat_(0,2)) * depth / cam_mat_(0,0);
  point.y() = (v - cam_mat_(1,2)) * depth / cam_mat_(1,1);
  point.z() = depth;

  return point;
}

Evaluator::ContainmentWindow getContainmentWindow(std::vector<float> distances, float pct) {
  float min = distances.front();
  float max = distances.back();
  
  // binary search for central X% stuff
  float max_dist = max;
  float min_dist = 0.0f;
  float EPSILON = 1e-3;
  float distance = 0.0f;
  
  Evaluator::ContainmentWindow window;
  window.pct = pct;
  std::vector<float> filtered;

  std::vector<float> positive;
  std::copy_if (distances.begin(), distances.end(), std::back_inserter(positive), [](float d){ return d > 0;} );

  while(max_dist - min_dist > EPSILON) {
    filtered.clear();
    distance = min_dist + (max_dist - min_dist) / 2.0f;
    std::copy_if (positive.begin(), positive.end(), std::back_inserter(filtered), [distance](float d){ return d < distance;} );
    if (filtered.size() > positive.size() * pct + EPSILON) {
      max_dist = distance;
    } else if (filtered.size() < positive.size() * pct - EPSILON) {
      min_dist = distance;
    } else {
      break;
    }
  }

  window.pos_bound = distance;

  max_dist = 0.0f;
  min_dist = min;
  filtered.clear();

  std::vector<float> negative;
  std::copy_if (distances.begin(), distances.end(), std::back_inserter(negative), [](float d){ return d < 0;} );

  while(max_dist - min_dist > EPSILON) {
    filtered.clear();
    distance = min_dist + (max_dist - min_dist) / 2.0f;
    std::copy_if (negative.begin(), negative.end(), std::back_inserter(filtered), [distance](float d){ return d > distance;} );

    if (filtered.size() > negative.size() * pct + EPSILON) {
      min_dist = distance;
    } else if (filtered.size() < negative.size() * pct - EPSILON) {
      max_dist = distance;
    } else {
      break;
    }
  }
  window.neg_bound = distance;

  return window;
}

Evaluator::ErrorHistogram getDistanceErrorHistogram(std::vector<float> error_distances, float bucket_size, bool compute_windows) {
  std::sort(error_distances.begin(), error_distances.end());
  float min = error_distances.front();
  float max = error_distances.back();

  Evaluator::ErrorHistogram histogram;
  size_t num_buckets = ceil((max - min) / bucket_size);
  histogram.buckets.resize(num_buckets);

  float lower = min;
  for(size_t bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
    histogram.buckets[bucket_idx].lower = lower;
    histogram.buckets[bucket_idx].upper = lower + bucket_size;
    histogram.buckets[bucket_idx].count = 0;
    lower += bucket_size;
  }

  size_t bucket_idx = 0;
  for(size_t i = 0; i < error_distances.size(); i++) {
    while (error_distances[i] >= histogram.buckets[bucket_idx].upper) {
      bucket_idx++;
    }

    histogram.buckets[bucket_idx].count++;
  }
  
  if (compute_windows) {
    for (float pct : Evaluator::PCT_WINDOWS) {
      Evaluator::ContainmentWindow window = getContainmentWindow(error_distances, pct);
      histogram.windows.push_back(window);
    }
  }

  return histogram;
}

Evaluator::ErrorHistogram Evaluator::getAbsoluteDistanceErrorHistogram() {
  return getDistanceErrorHistogram(dist_errors_, 0.5f, true);
}

Evaluator::ErrorHistogram Evaluator::getRelativeDistanceErrorHistogram() {
  return getDistanceErrorHistogram(rel_dist_errors_, 0.1f, true);
}

Evaluator::ErrorHistogram Evaluator::getErrorTrackSizeHistogram() {
  std::vector<float> sizes(error_tracks_.size());
  size_t n = 0;
  std::generate(sizes.begin(), sizes.end(), [this, n] () mutable { return error_tracks_[n++].loc_map_history.size(); });
  return getDistanceErrorHistogram(sizes, 1.0f, false);
}

} // namespace IVOA


