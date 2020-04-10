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

#include "dataset.h"


namespace IVOA {

using std::vector;
using std::string;
using std::cout;
using std::endl;
using cv::Point;



Dataset::Dataset(float patch_size,
                 int session_num,
                 std::string dataset_dir,
                 double obstacle_ratio_thresh,
                 float distance_err_thresh,
                 float rel_distance_err_thresh,
                 float max_range):
        patch_size_(patch_size),
        session_num_(session_num),
        dataset_dir_(dataset_dir),
        obstacle_ratio_thresh_(obstacle_ratio_thresh),
        distance_err_thresh_ (distance_err_thresh),
        rel_distance_err_thresh_(rel_distance_err_thresh),
        max_range_(max_range){
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(5) << session_num_;
  session_name_ = ss.str(); 
  
  PrepareDirectories();
}

void Dataset::LoadQueryPoints(const std::vector<cv::Point>& query_points) {
  query_points_ = query_points;
  query_points_loaded_ = true;
}

bool Dataset::LabelData(const cv::Mat& gt_obstacle_img,
                        const cv::Mat& gt_obstacle_dist,
                        const cv::Mat& pred_obstacle_img,
                        const cv::Mat& pred_obstacle_dist,
                        const std::string& name_prefix) {
  if (!query_points_loaded_) {
    return false;
  }
 
  ExtractPatchLabels(gt_obstacle_img,
                     gt_obstacle_dist,
                     &gt_labels_,
                     &gt_obstacle_distance_,
                     &gt_valid_pts_,
                     max_range_);
  ExtractPatchLabels(pred_obstacle_img,
                     pred_obstacle_dist,
                     &pred_labels_,
                     &pred_obstacle_distance_,
                     &pred_valid_pts_);
  
  // Populate valid patches flags given gt_valid_pts_ and pred_valid_pts_
  int valid_pts_count = 0;
  valid_pts_.clear();
  for (size_t i = 0; i < gt_valid_pts_.size(); i++) {
    valid_pts_.push_back(gt_valid_pts_[i] && pred_valid_pts_[i]);
    if (valid_pts_.back()) {
      valid_pts_count++;
    }
  }
  
  patch_coord_ = query_points_;
 
  // Labels the patch as either TP (0), TN (1), FP (2), and FN (3)
  GenerateMultiClassLabels();
  
  latest_img_name_prefix_ = name_prefix;
  
  UpdateDataset();
  
  img_count_++;
  img_patches_count_+= valid_pts_count;
  return true;
}

void Dataset::SaveImages(const cv::Mat& left_im) {
  string left_im_out_dir = dataset_dir_ + session_name_ + "/" 
                          + kLeftCamFolder_;
  string left_annotated_im_out_dir = dataset_dir_ + session_name_ + "/" 
                          + kLeftCamAnnotatedFolder_;  
  string left_annotated_comp_im_out_dir = dataset_dir_ + session_name_ + "/" 
                          + kLeftCamAnnotatedComprehensiveFolder_; 
 
  cv::Mat annotated_left_im = AnnotateImage(left_im);
  cv::Mat annotated_comprehensive_left_im = AnnotateImageComprehensive(left_im);
 
  cv::imwrite(left_im_out_dir + left_img_name_, left_im);
  cv::imwrite(left_annotated_im_out_dir + left_img_annotated_name_, 
              annotated_left_im);
  cv::imwrite(left_annotated_comp_im_out_dir + left_img_annotated_comp_name_, 
              annotated_comprehensive_left_im);
}

void Dataset::SaveToFile() {
  string curr_out_dir = dataset_dir_ + "/" + session_name_ + "/";
  if(!WriteJsonToFile(curr_out_dir, 
                  kFullImageNamesFile_, 
                  json_val_full_img_names_)) {
    LOG(FATAL) << "Could not create directory "
              << curr_out_dir+kFullImageNamesFile_ ;                
  }
  
  if(!WriteJsonToFile(curr_out_dir,
                  kImagePatchDataFile_,
                  json_val_patch_data_)) {
    LOG(FATAL) << "Could not create directory "
              << curr_out_dir+kImagePatchDataFile_ ;
  }
  if(!WriteJsonToFile(curr_out_dir,
                  kImagePatchNamesFile_,
                  json_val_patch_names_)) {
    LOG(FATAL) << "Could not create directory "
              << curr_out_dir+kImagePatchNamesFile_ ;
  }
}

float Dataset::GetMinValueInMat(const cv::Mat& query_img,
                                const cv::Mat& mask) {
  float min_val = std::numeric_limits<float>::max();
  for (size_t i = 0; i < query_img.rows; i++) {
    for (size_t j = 0; j < query_img.cols; j++) {
      if (min_val > query_img.at<float>(i, j) &&
          mask.at<uint8_t>(i,j)) {
        min_val = query_img.at<float>(i, j);
      }
    }
  }
  
  return min_val;
}


void Dataset::UpdateDataset() {
  // Update the name of the image
  char buff_l[30];
  sprintf(buff_l, "%s_%s_l.jpg", session_name_.c_str(), 
                                  latest_img_name_prefix_.c_str());
  left_img_name_ = buff_l;
  
  char buff_ann_l[30];
  sprintf(buff_ann_l, "%s_%s_l_ann.jpg", session_name_.c_str(), 
                                  latest_img_name_prefix_.c_str());
  left_img_annotated_name_ = buff_ann_l;
  
  char buff_ann_comp_l[40];
  sprintf(buff_ann_comp_l, "%s_%s_l_ann_comp.jpg", session_name_.c_str(), 
                                  latest_img_name_prefix_.c_str());
  left_img_annotated_comp_name_ = buff_ann_comp_l;

  
  // Update the name of the image patches
  patch_l_names_.clear();
  for (size_t i = 0; i < patch_coord_.size(); i++) {
    char buff_l_p[30];
    sprintf(buff_l_p, "%s_%s_%05lu_l.jpg", session_name_.c_str(),
            latest_img_name_prefix_.c_str(), i);
    string patch_l_name = buff_l_p;

    patch_l_names_.push_back(patch_l_name);
  }
  
  AppendToJsonFullImNames();
  AppendToJsonImPatchNames();
  AppendToJsonImPatchData();
}




// Function to append data to the JSON file
// void Dataset::AppendToJsonFullImData(double time_sec) {
//   if(!json_val_full_img_data_.isMember("bagfile_num")) {
//     json_val_full_img_data_["bagfile_num"].append(std::stoi(session_name_));
//   }
//   json_val_full_img_data_["time_stamp"].append(time_sec);
// }

void Dataset::AppendToJsonFullImNames() {
  if(!json_val_full_img_names_.isMember("bagfile_name")) {
    json_val_full_img_names_["bagfile_name"].append(session_name_);
  }
  json_val_full_img_names_["img_names_left"].append(left_img_name_);

  Json::Value patch_names_obj;
  Json::Value patch_ind_obj;
  int patch_idx = 0;
  for (size_t i = 0; i < patch_l_names_.size(); i++) {
    if (!gt_valid_pts_[i] || !pred_valid_pts_[i]) {
      continue;
    }
    patch_names_obj["names"].append(patch_l_names_[i]);
    patch_ind_obj["indices"].append((int)(img_patches_count_) + patch_idx);
    patch_idx++;
  }
  json_val_full_img_names_["corr_img_patch_names"].append(patch_names_obj);
  json_val_full_img_names_["corr_img_patch_indices"].append(patch_ind_obj);
}

void Dataset::AppendToJsonImPatchData() {
  if(!json_val_patch_data_.isMember("bagfile_num")) {
    json_val_patch_data_["bagfile_num"].append(stoi(session_name_));
  }
  
  // TODO: modify the names for obstacle existence
  for (size_t i = 0; i < patch_coord_.size(); i++) {
    if (!gt_valid_pts_[i] || !pred_valid_pts_[i]) {
      continue;
    }
    json_val_patch_data_["patch_coordinate_left"]["x"].append(
              patch_coord_[i].x);
    json_val_patch_data_["patch_coordinate_left"]["y"].append(
              patch_coord_[i].y);
    
    bool pred_label = pred_labels_[i];
    bool gt_label = gt_labels_[i];
    float pred_obs_dist = pred_obstacle_distance_[i];
    float gt_obs_dist = gt_obstacle_distance_[i];

    json_val_patch_data_["jpp_obs_existence"].append(pred_label);
    json_val_patch_data_["jpp_obs_distance"].append(pred_obs_dist);
    json_val_patch_data_["kinect_obs_existence"].append(gt_label);
    json_val_patch_data_["kinect_obs_distance"].append(gt_obs_dist);
    json_val_patch_data_["multi_class_label"].append(multi_class_labels_[i]);
  }
}

void Dataset::AppendToJsonImPatchNames() {
  if(!json_val_patch_names_.isMember("bagfile_name")) {
    json_val_patch_names_["bagfile_name"].append(session_name_);
  }

  for (size_t i = 0; i < patch_l_names_.size(); i++) {
    if (!gt_valid_pts_[i] || !pred_valid_pts_[i]) {
      continue;
    }
    json_val_patch_names_["img_patch_name"].append(patch_l_names_[i]);
    json_val_patch_names_["corr_full_img_name"].append(left_img_name_);
    json_val_patch_names_["corr_full_img_index"].append(
                          atoi(latest_img_name_prefix_.c_str()));
  }
}

// Returns false if the patch is not valid (out of bounds)
bool Dataset::LabelPatch(const cv::Mat& obstacle_img,
                         const cv::Mat &obstacle_dist,
                         const cv::Point& patch_coord, 
                         const float& patch_size, 
                         const float& obstacle_ratio_thresh,
                         bool *label,
                         float *patch_obs_dist,
                         float max_range) {
  cv::Size patch_size2d(static_cast<float>(patch_size_), 
                      static_cast<float>(patch_size_)); 
    
  if (!ValidatePatch(obstacle_img.size(),
                            patch_size2d,
                            patch_coord)) {
    *label = true;
    return false;
  }
  
    
  cv::Rect patch_def(patch_coord.x - patch_size/2.0,
                patch_coord.y - patch_size/2.0,
                patch_size,
                patch_size);
  cv::Mat patch = obstacle_img(patch_def);
  cv::Mat patch_dist = obstacle_dist(patch_def);
  double sum = cv::sum(patch)[0];
  double obs_ratio = sum / static_cast<double>(patch_size_ * patch_size_);
  
  
  if (obs_ratio > obstacle_ratio_thresh) {
    *label = true;
  } else {
    *label = false;
  }
  
  // Get the minimum distance obstacle in the given image patch
  *patch_obs_dist = GetMinValueInMat(patch_dist, patch);
  
 
  // If there is a required max range, the patch will be flagged as invalid
  // for training if the closest point in the patch (regardless of the point
  // being an obstacle or not) is farther than max_range or if the patch is 
  // labeled as an obstacle, it will be flagged as invalid when 
  // dist_to_obstacle is larger than max_range
  if (max_range >= 0) {
    // The minimum distance to any 3D point associated with the pixels in the 
    // patch (the point does not need to be an obstacle)
    float min_dist = GetMinValueInMat(patch_dist, 
                            cv::Mat::ones(patch_size, patch_size,CV_8U));
    if (min_dist > max_range || 
        (*label && *patch_obs_dist > max_range)) {
      return false;
    }
  }
  
 
  return true;
}

void Dataset::ExtractPatchLabels(const cv::Mat& obstacle_img,
                                 const cv::Mat &obstacle_dist,
                                std::vector<bool> *obs_existence,
                                std::vector<float> *obs_distance,
                                std::vector<bool> *valid_pts,
                                float max_range) {
  obs_existence->clear();
  obs_distance->clear();
  valid_pts->clear();
  for (size_t i = 0; i < query_points_.size(); i++) {
    bool label;
    // The minimum distance to any obstacles in the image patch
    float patch_obs_dist;
    bool valid = LabelPatch(obstacle_img,
                            obstacle_dist,
                            query_points_[i],
                            patch_size_,
                            obstacle_ratio_thresh_,
                            &label,
                            &patch_obs_dist,
                            max_range);
    obs_existence->push_back(label);
    obs_distance->push_back(patch_obs_dist);
    valid_pts->push_back(valid);
  }
}

void Dataset::GenerateMultiClassLabels() {
  // TODO: For the case when an obstacle is in the form of a hole 
  // (z_obs < 0), you do not care about the accuracy of the depth estimate
  // but only about the existence of one
  multi_class_labels_.clear();
  for (size_t i = 0; i < patch_coord_.size(); i++){
    if ((gt_labels_[i] == pred_labels_[i]) && gt_labels_[i]) {
      float obst_dist_err = gt_obstacle_distance_[i] - 
                            pred_obstacle_distance_[i];
      float rel_err_thresh = rel_distance_err_thresh_ * 
                             gt_obstacle_distance_[i]; 
      if (obst_dist_err > std::max(distance_err_thresh_, rel_err_thresh)) {
        // False Positive
        multi_class_labels_.push_back(FP);
      } else if (
           obst_dist_err < -std::max(distance_err_thresh_, rel_err_thresh)) {
        // Flase Negative
        multi_class_labels_.push_back(FN);
      } else {
        // True Positive
        multi_class_labels_.push_back(TP);
      }
    } else if ((gt_labels_[i] != pred_labels_[i]) && gt_labels_[i]) {
      // False negative
      multi_class_labels_.push_back(FN);
    } else if ((gt_labels_[i] == pred_labels_[i]) && !gt_labels_[i]) {
      // True negative
      multi_class_labels_.push_back(TN);
    } else {
      // False positive
      multi_class_labels_.push_back(FP);
    }
  }
}


bool Dataset::ValidatePatch(const cv::Size& img_size, 
                            const cv::Size& patch_size, 
                            const cv::Point& patch_coord) {
  if (patch_coord.x + patch_size.width/2.0 > img_size.width ||
      patch_coord.x - patch_size.width/2.0 < 0 ||
      patch_coord.y + patch_size.height/2.0 > img_size.height ||
      patch_coord.y - patch_size.height/2.0 < 0) {
        return false;
  } else {
    return true;
  }
}

void Dataset::PrepareDirectories() {
  RemoveDirectory(dataset_dir_ + session_name_ + "/" + kImageFolder_);
  CreateDirectory(dataset_dir_);
  CreateDirectory(dataset_dir_ + session_name_);

  CreateDirectory(dataset_dir_ + session_name_ + "/" + kImageFolder_);
  CreateDirectory(dataset_dir_ + session_name_ + "/" + kLeftCamFolder_);
  CreateDirectory(dataset_dir_ + session_name_ + "/" +
                  kLeftCamPatchesFolder_);
  CreateDirectory(dataset_dir_ + session_name_ + "/" + 
                  kLeftCamAnnotatedFolder_);
  CreateDirectory(dataset_dir_ + session_name_ + "/" + 
                  kLeftCamAnnotatedComprehensiveFolder_);
}

cv::Mat Dataset::AnnotateImage(const cv::Mat &image) {
  const int radius_gt = 7;
  const int radius_pred = 4;
  
  cv::Scalar green = cv::Scalar(0, 255 , 0);
  cv::Scalar red = cv::Scalar(0, 0 , 255);
  
  cv::Mat annotated_img = image.clone();
  for (size_t i = 0; i < patch_coord_.size(); i++){
    if (!gt_valid_pts_[i] || !pred_valid_pts_[i]) {
      continue;
    }
    cv::Scalar gt_color = (gt_labels_[i])? red : green;
    cv::Scalar pred_color = (pred_labels_[i])? red : green;
    
    cv::circle(annotated_img, patch_coord_[i], radius_gt, gt_color, -1, 8, 0);
    cv::circle(annotated_img, 
               patch_coord_[i], 
               radius_pred, 
               pred_color, 
               -1, 8, 0);
  }
  
  return annotated_img;
}

cv::Mat Dataset::AnnotateImageComprehensive(const cv::Mat &image) {
  const int radius_label = 10;
  const int radius_gt = 7;
  const int radius_pred = 4;
  
  cv::Scalar green = cv::Scalar(0, 255 , 0);
  cv::Scalar red = cv::Scalar(0, 0 , 255);
  cv::Scalar orange = cv::Scalar(0, 165 , 255);
  cv::Scalar blue = cv::Scalar(255, 0 , 0);
  
  cv::Mat annotated_img = image.clone();
  for (size_t i = 0; i < patch_coord_.size(); i++){
    if (!gt_valid_pts_[i] || !pred_valid_pts_[i]) {
      continue;
    }
    cv::Scalar gt_color = (gt_labels_[i])? red : green;
    cv::Scalar pred_color = (pred_labels_[i])? red : green;
    
    cv::Scalar label_color;
    // Label the patch as either TP (green), TN (blue), FP (orange), and FN(red)
    if ((gt_labels_[i] == pred_labels_[i]) && gt_labels_[i]) {
      float obst_dist_err = gt_obstacle_distance_[i] - 
                            pred_obstacle_distance_[i];
      float rel_err_thresh = rel_distance_err_thresh_ * 
                             gt_obstacle_distance_[i]; 
      if (obst_dist_err > std::max(distance_err_thresh_, rel_err_thresh)) {
        // False Positive
        label_color = orange;
      } else if (
         obst_dist_err < -std::max(distance_err_thresh_, rel_err_thresh)) {
        // Flase Negative
        label_color = red;
      } else {
        // True Positive
        label_color = green;
      }
    } else if ((gt_labels_[i] != pred_labels_[i]) && gt_labels_[i]) {
      // False negative
      label_color = red;
    } else if ((gt_labels_[i] == pred_labels_[i]) && !gt_labels_[i]) {
      // True negative
      label_color = blue;
    } else {
      // False positive
      label_color = orange;
    }
    
    cv::circle(annotated_img, 
               patch_coord_[i], 
               radius_label, 
               label_color, 
               -1, 8, 0);
    cv::circle(annotated_img, patch_coord_[i], radius_gt, gt_color, -1, 8, 0);
    cv::circle(annotated_img, 
               patch_coord_[i], 
               radius_pred, 
               pred_color, 
               -1, 8, 0);
   
  }
  
  // Legend
  cv::Mat roi = annotated_img(cv::Rect(0, 550, 200, 50));
  cv::Mat color(roi.size(), CV_8UC4, cv::Scalar(125, 125, 125,255)); 
  double alpha = 0.7;
  cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi); 
  
  int baseline = 0;
  double font_scale = 0.8;
  float y_offset = 550 + 5;
  float x_offset = 50;
  float x_start = 20;
  
  
  vector<string> text = {"TP", "TN", "FP", "FN"};
  vector<cv::Scalar> text_color = {green, blue, orange, red};
  
  for (size_t i = 0; i < text.size(); i++) {
    cv::Size txt_size = getTextSize(text[i], 
                            cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
    
    cv::Point2f cursor((i) *x_offset + x_start,  y_offset);
    cursor.x -= txt_size.width/2;
    cursor.y += txt_size.height + baseline;
    putText(annotated_img, 
            text[i], 
            cursor, 
              cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color[i], 3);
      
  }
  

  
  return annotated_img;
}


} // namespace IVOA

