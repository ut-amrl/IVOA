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

#ifndef IVOA_DATASET_
#define IVOA_DATASET_

#include <vector>
#include <algorithm>
#include <glog/logging.h>
#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <dirent.h>
#include <jsoncpp/json/json.h>
#include <boost/filesystem.hpp>


#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <bitset>
#include <cstdio>

#include "io_access.h"


namespace IVOA {
class Dataset{
public:
  Dataset(float patch_size,
          int session_num,
          std::string dataset_dir,
          double obstacle_ratio_thresh);
  ~Dataset() = default;
  
  void LoadQueryPoints(const std::vector<cv::Point> &query_points);
  bool LabelData(const cv::Mat &gt_obstacle_img,
                 const cv::Mat &pred_obstacle_img,
                 const std::string& name_prefix);
  
  void SaveImages(const cv::Mat& left_im);
  
  // Saves all data to file
  void SaveToFile();
  
private:
  // Adds the new extracted patches and labels to the dataset
  void UpdateDataset();
  
  void AppendToJsonFullImNames();
  void AppendToJsonImPatchData();
  void AppendToJsonImPatchNames();
  
  
  // Given an obstacle image (boolean value for each pixel where 1 means 
  // existence of an obstacle) and a patch size and coordinate, outputs 
  // a label for that patch as either obstacle or traversable. If more than
  // obstacle_perc_thresh percent of the pixels are obstacle the patch will
  // be labeled as obstacle
  bool LabelPatch(const cv::Mat &obstacle_img,
                  const cv::Point &patch_coord,
                  const float &patch_size,
                  const float &obstacle_ratio_thresh,
                  bool *label);
  
  // Extracts patches and labels them as obstacle or traversable for a given
  // obstacle image
  void ExtractPatchLabels(const cv::Mat &obstacle_img,
                          std::vector<bool> *obs_existence,
                          std::vector<bool> *valid_pts);
  
  bool ValidatePatch(const cv::Size &img_size,
                     const cv::Size &patch_size,
                     const cv::Point &patch_coord);
  
  void PrepareDirectories();
  
  // Annotates the input image with the currently extracted labels (both 
  // predicted and ground truth)
  cv::Mat AnnotateImage(const cv::Mat &image);
  
          
  
  bool query_points_loaded_ = false;
  float patch_size_;
  long unsigned int img_count_ = 0;
  long int img_patches_count_ = 0;
  int session_num_;
  std::string latest_img_name_prefix_;
  std::string session_name_;
  std::string dataset_dir_;
  
  // If more than obstacle_perc_thresh percent of the pixels in a patch are 
  // obstacle the patch will be labeled as obstacle
  double obstacle_ratio_thresh_;
  
  std::string session_id_;
  std::vector<cv::Point> query_points_;
  
  // Holds the ground truth labels for all query points on the latest loaded 
  // image
  std::vector<bool> gt_labels_;
  std::vector<bool> gt_valid_pts_;
  
  // Holds the predicted labels for all query points on the latest loaded 
  // image
  std::vector<bool> pred_labels_;
  std::vector<bool> pred_valid_pts_;
  
  // Coordinate of the extracted and pruned patches
  std::vector<cv::Point> patch_coord_;
  
  // The name for the latest image
  std::string left_img_name_;
  
  std::string left_img_annotated_name_;
  
  // The names for all extracted patches from the latest loaded image
  std::vector<std::string> patch_l_names_;
  
  Json::Value json_val_full_img_data_;
  Json::Value json_val_full_img_names_;
  Json::Value json_val_patch_data_;
  Json::Value json_val_patch_names_;
  
  // Constants
  const std::string kImageFolder_ = "images/";
  const std::string kLeftCamFolder_ = "images/left_cam/";
  const std::string kLeftCamPatchesFolder_ = "images/left_cam_patch/";
  const std::string kLeftCamAnnotatedFolder_ = "images/left_cam_annotated/";
  
  // This file stores the corresponding data for each full image
  const std::string kFullImageDataFile_ = "full_images_data.json";
  const std::string kFullImageNamesFile_ = "full_images_names.json";
  
  // This file stores the corresponding data for each image patch
  const std::string kImagePatchDataFile_ = "image_patches_data.json";
  const std::string kImagePatchNamesFile_ = "image_patches_names.json";
  
  
};
} // namespace IVOA

#endif // IVOA_DATASET_



