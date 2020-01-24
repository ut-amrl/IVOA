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
                 double obstacle_ratio_thresh):
        patch_size_(patch_size),
        session_num_(session_num),
        dataset_dir_(dataset_dir),
        obstacle_ratio_thresh_(obstacle_ratio_thresh){
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
                        const cv::Mat& pred_obstacle_img) {
  if (!query_points_loaded_) {
    return false;
  }
  
  // TODO: prune the patch labels 
  ExtractPatchLabels(gt_obstacle_img,
                     &gt_labels_,
                     &gt_valid_pts_);
  ExtractPatchLabels(pred_obstacle_img,
                     &pred_labels_,
                     &pred_valid_pts_);
  
  //TODO: Prune patch labels:
  patch_coord_ = query_points_;
  
  UpdateDataset();
  
  img_count_++;
  img_patches_count_+=gt_labels_.size();
  return true;
}

void Dataset::SaveImages(const cv::Mat& left_im) {
  string left_im_out_dir = dataset_dir_ + session_name_ + "/" 
                          + kLeftCamFolder_;
  string left_annotated_im_out_dir = dataset_dir_ + session_name_ + "/" 
                          + kLeftCamAnnotatedFolder_;                          
 
  cv::Mat annotated_left_im = AnnotateImage(left_im);
 
  cv::imwrite(left_im_out_dir + left_img_name_, left_im);
  cv::imwrite(left_annotated_im_out_dir + left_img_annotated_name_, 
              annotated_left_im);
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


void Dataset::UpdateDataset() {
  // Update the name of the image
  char buff_l[30];
  sprintf(buff_l, "%s_%010lu_l.jpg", session_name_.c_str(), img_count_);
  left_img_name_ = buff_l;
  
  char buff_ann_l[30];
  sprintf(buff_ann_l, "%s_%010lu_l_ann.jpg", session_name_.c_str(),img_count_);
  left_img_annotated_name_ = buff_ann_l;

  
  // Update the name of the image patches
  patch_l_names_.clear();
  for (size_t i = 0; i < patch_coord_.size(); i++) {
    char buff_l_p[30];
    sprintf(buff_l_p, "%s_%010lu_%05lu_l.jpg", session_name_.c_str(),
            img_count_, i);
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
  for (size_t i = 0; i < patch_l_names_.size(); i++) {
    patch_names_obj["names"].append(patch_l_names_[i]);
    patch_ind_obj["indices"].append((int)(img_patches_count_ + i));
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
    json_val_patch_data_["patch_coordinate_left"]["x"].append(
              patch_coord_[i].x);
    json_val_patch_data_["patch_coordinate_left"]["y"].append(
              patch_coord_[i].y);
    
    bool pred_label = pred_labels_[i];
    bool gt_label = gt_labels_[i];

    json_val_patch_data_["jpp_obs_existence"].append(pred_label);
    json_val_patch_data_["kinect_obs_existence"].append(gt_label);
  }
}

void Dataset::AppendToJsonImPatchNames() {
  if(!json_val_patch_names_.isMember("bagfile_name")) {
    json_val_patch_names_["bagfile_name"].append(session_name_);
  }

  for (size_t i = 0; i < patch_l_names_.size(); i++) {
    json_val_patch_names_["img_patch_name"].append(patch_l_names_[i]);
    json_val_patch_names_["corr_full_img_name"].append(left_img_name_);
    json_val_patch_names_["corr_full_img_index"].append(
                          (int)(img_count_));
  }
}

// Returns false if the patch is not valid (out of bounds)
bool Dataset::LabelPatch(const cv::Mat& obstacle_img, 
                         const cv::Point& patch_coord, 
                         const float& patch_size, 
                         const float& obstacle_ratio_thresh,
                         bool *label) {
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
  double sum = cv::sum(patch)[0];
  double obs_ratio = sum / static_cast<double>(patch_size_ * patch_size_);
  
  
  if (obs_ratio > obstacle_ratio_thresh) {
    *label = true;
  } else {
    *label = false;
  }
 
  return true;
}

void Dataset::ExtractPatchLabels(const cv::Mat& obstacle_img,
                                std::vector<bool> *obs_existence,
                                std::vector<bool> *valid_pts) {
  obs_existence->clear();
  valid_pts->clear();
  for (size_t i = 0; i < query_points_.size(); i++) {
    bool label;
    bool valid = LabelPatch(obstacle_img,
                            query_points_[i],
                            patch_size_,
                            obstacle_ratio_thresh_,
                            &label);
    obs_existence->push_back(label);
    valid_pts->push_back(valid);
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
}

cv::Mat Dataset::AnnotateImage(const cv::Mat &image) {
  const int radius_gt = 7;
  const int radius_pred = 4;
  
  cv::Scalar green = cv::Scalar(0, 255 , 0);
  cv::Scalar red = cv::Scalar(0, 0 , 255);
  
  cv::Mat annotated_img = image.clone();
  for (size_t i = 0; i < patch_coord_.size(); i++){
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


} // namespace IVOA

