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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/console.h>
#include <ros/ros.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "yaml-cpp/yaml.h"
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "cnpy.h"
#include "dataset.h"
#include "depth2pointcloud.h"
#include "io_access.h"

using cv::Mat;
using namespace std;
using namespace IVOA;
using cv::Point;
using std::vector;

// Command line flags flag
DEFINE_int32(session_num, 0,
             "Session number to identify the generated "
             "portion of the dataset.");
DEFINE_string(source_dir, "",
              "Path to the base directory of the source "
              "dataset.");
DEFINE_string(source_dir_predictions, "",
              "Path to the base directory of "
              "the predictions. If empty, source_dir is used.");
DEFINE_string(cam_extrinsics_path, "",
              "Path to the file containing the "
              "left camera calibration file.");
DEFINE_string(output_dataset_dir, "", "Path to save the generated datatset. ");
DEFINE_string(pred_depth_fmt, "npy",
              "Format of the predicted depth images."
              " Select between {npy, pfm}");
DEFINE_string(pred_depth_folder, "img_left",
              "Name of the folder under the source_dir where predicted "
              "depth images are stored.");
DEFINE_double(margin_width, 5,
              "Margin around the edge of the images to throw "
              " out during evaluation. Value is interpreted as the percentage "
              "of the image"
              " width");
DEFINE_double(patch_size, 50,
              "Size of image patches to be labeled for "
              "training of IVOA.");
DEFINE_double(patch_stride, 30,
              "The stride size in pixels between two patches.");
DEFINE_double(
    positive_height_obs_thresh, 0.3,
    "Minimum height of an obstacle in the positive direction of z axis.");
DEFINE_double(negative_height_obs_thresh, 0.3,
              "Minimum height of an obstacle in the negative direction of the "
              "z axis (minimum depth of a hole in the ground).");
DEFINE_double(image_width, 960, "Input image width.");
DEFINE_double(image_height, 600, "Input image height.");

// If the predicted distance to obstacle is off from the ground truth by
// more than max(FLAGS_distance_err_thresh, kRelativeErrThresh * TrueDistance),
// it will be labeled as either FP (if pred_dist < gt_dist) or FN (if pred_dist
// > gt_dist)
DEFINE_double(distance_err_thresh, 1.0,
              "Absolute distance error threshold for a prediction to be "
              "considered erroneous (meters).");
DEFINE_double(relative_distance_err_thresh, 0.1,
              "Relative distance error threshold for a prediction to be "
              "considered erroneous.");
DEFINE_double(min_patch_info_ratio, 0.1,
              "The minimum ratio of available pixel readings to the total" 
              " number of pixels in an image patch in order to consider" 
              " existence of enough information.");
DEFINE_double(min_err_pixel_ratio, 0.1,
              "The minimum ratio of pixels with depth estimation error to the" 
              " total number of pixels in an image patch in order to label"
              " the patch as an instance of depth error (either FP or FN)." 
              " NOTE: This is only used when is_pixel_wise_mode_ is true");

DECLARE_bool(help);
DECLARE_bool(helpshort);

// Objects that are further than max_range_ away from the agent will not be
// considered for obstacle avoidance. A negative value implies that the
// max_range constraint will not be enforced
DEFINE_double(max_range, 50.0,
              "Max range to consider for depth prediction evaluation");

// Depth readings smaller than min_range will be ignored. Values smaller than
// min_range_ in the reference depth image indicates GT depth information
// being unavailable for those pixels.
DEFINE_double(min_range, 0.01,
              "Min range to consider for depth prediction evaluation");

DEFINE_bool(
    visualization, false,
    "Whether or not to publish visualization information while executing.");
DEFINE_bool(
    debug, false,
    "Whether or not run with debugging visualizations and print statements.");
DEFINE_bool(flip_depth_images, false,
            "Whether or not to flip the depth images vertically. This is required for AirSim data.");
DEFINE_bool(generate_labels_in_pixel_wise_mode, false,
            "Whether or not to generate labels in pixel wise mode.");

// Parameters
const double kObstacleRatioThresh = 0.05;

// Checks if all required command line arguments have been set
void CheckCommandLineArgs(char **argv) {
  vector<string> required_args = {"session_num", "source_dir",
                                  "cam_extrinsics_path", "output_dataset_dir"};

  for (const string &arg_name : required_args) {
    bool flag_not_set =
        gflags::GetCommandLineFlagInfoOrDie(arg_name.c_str()).is_default;
    if (flag_not_set) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "main_airsim_processing");
      LOG(FATAL) << arg_name << " was not set." << endl;
    }
  }
}

vector<Point> GenerateQueryPoints(const cv::Size img_size,
                                  const float &patch_size, const float &stride,
                                  const float &margin_width) {
  vector<Point> query_points;
  float x = 0;
  float y = 0;

  for (x = patch_size / 2 + margin_width;
       x + patch_size / 2 < (img_size.width - margin_width); x += stride) {
    for (y = patch_size / 2 + margin_width;
         y + patch_size / 2 < (img_size.height - margin_width); y += stride) {
      query_points.push_back(Point(x, y));
    }
  }

  return query_points;
}

int main(int argc, char **argv) {
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO level logging.
  FLAGS_colorlogtostderr = 1; // Colored logging.
  FLAGS_logtostderr = true;   // Don't log to disk

  string usage(
      "This program converts the recorded data from AirSim along with"
      " the predicted depth output of the monodepth to the IVOA dataset "
      " format :\n");
  usage += string(argv[0]) + " <argument1> <argument2> ...";
  gflags::SetUsageMessage(usage);

  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_help) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "main_airsim_processing");
    return 0;
  }
  CheckCommandLineArgs(argv);

  ros::init(argc, argv, "IVOA_data_preprocessing",
            ros::init_options::NoSigintHandler);
  ros::NodeHandle nh;

  ros::Publisher point_cloud_publisher_gt =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/gt_pointcloud", 1);
  ros::Publisher point_cloud_publisher_pred =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/pred_pointcloud", 1);
  ros::Publisher point_cloud_publisher_gt_filt_dist =
      nh.advertise<sensor_msgs::PointCloud2>(
          "/ivoa/gt_pointcloud_filtered_by_dist", 1);
  ros::Publisher point_cloud_publisher_pred_filt_height =
      nh.advertise<sensor_msgs::PointCloud2>(
          "/ivoa/pred_pointcloud_filtered_by_height", 1);

  std::cout << "Processing data on patches of size " << FLAGS_patch_size
            << std::endl;

  // Using only one instance of Depth2Pointcloud since the depth and left RGB
  // camera share the same extrinsics and intrinsics in our AirSim setup
  Depth2Pointcloud depth_img_converter;
  depth_img_converter.LoadCameraCalibration(FLAGS_cam_extrinsics_path);

  if (FLAGS_source_dir_predictions.empty()) {
    FLAGS_source_dir_predictions = FLAGS_source_dir;
  }
  std::cout << "Loading data from " << FLAGS_source_dir << std::endl;
  std::cout << "Loading predictions from " << FLAGS_source_dir_predictions
            << std::endl;

  string gt_depth_dir = FLAGS_source_dir + "/img_depth/";
  string pred_depth_dir =
      FLAGS_source_dir_predictions + "/" + FLAGS_pred_depth_folder + "/";
  string left_img_dir = FLAGS_source_dir + "/img_left/";

  // Read the prefix of all avaiable datapoints
  vector<int> filename_prefixes;
  GetFileNamePrefixes(pred_depth_dir, &filename_prefixes);
  std::sort(filename_prefixes.begin(), filename_prefixes.end());

  Dataset dataset(FLAGS_patch_size,
                  FLAGS_session_num,
                  FLAGS_output_dataset_dir,
                  kObstacleRatioThresh,
                  FLAGS_distance_err_thresh,
                  FLAGS_relative_distance_err_thresh,
                  FLAGS_max_range,
                  FLAGS_min_range,
                  FLAGS_generate_labels_in_pixel_wise_mode,
                  FLAGS_min_patch_info_ratio,
                  FLAGS_min_err_pixel_ratio);

  cv::Size image_size(FLAGS_image_width, FLAGS_image_height);
  vector<Point> query_points = GenerateQueryPoints(
      image_size, FLAGS_patch_size, FLAGS_patch_stride, FLAGS_margin_width);

  dataset.LoadQueryPoints(query_points);

  const bool kSkipFirstFrame = true;
  int count = 0;
  for (const int &i : filename_prefixes) {

    if (kSkipFirstFrame && (count == 0)) {
      count++;
      continue;
    }

    stringstream ss;
    ss << setfill('0') << setw(10) << i;
    string prefix = ss.str();
    string gt_depth_path = gt_depth_dir + prefix + ".pfm";
    string left_img_path = left_img_dir + prefix + ".png";
    string pred_depth_path;
    if (FLAGS_pred_depth_fmt == "npy") {
      pred_depth_path = pred_depth_dir + prefix + "_disp.npy";
    } else if (FLAGS_pred_depth_fmt == "pfm") {
      pred_depth_path = pred_depth_dir + prefix + ".pfm";
    } else {
      LOG(FATAL) << "Unknown predicted depth image format "
                 << FLAGS_pred_depth_fmt;
    }

    // Read the left cam image
    Mat left_img = cv::imread(left_img_path, CV_LOAD_IMAGE_UNCHANGED);
    if (left_img.channels() == 1) {
      cvtColor(left_img, left_img, cv::COLOR_GRAY2RGBA);
    }

    // Read the Ground truth depth image
    PFM pfm_rw;
    float *depth_data = pfm_rw.read_pfm<float>(gt_depth_path);
    cv::Mat depth_img_gt =
        cv::Mat(pfm_rw.getHeight(), pfm_rw.getWidth(), CV_32F, depth_data);
    if (FLAGS_flip_depth_images) {
      cv::flip(depth_img_gt, depth_img_gt, 0);
    }

    // Read the predicted depth image
    cv::Mat depth_img_pred;
    if (FLAGS_pred_depth_fmt == "npy") {
      cnpy::NpyArray arr = cnpy::npy_load(pred_depth_path);
      float *loaded_data = arr.data<float>();
      depth_img_pred = cv::Mat(arr.shape[0], arr.shape[1], CV_32F, loaded_data);
    } else if (FLAGS_pred_depth_fmt == "pfm") {
      PFM pfm_rw_p;
      float *pred_depth_data = pfm_rw_p.read_pfm<float>(pred_depth_path);
      depth_img_pred = cv::Mat(pfm_rw_p.getHeight(), pfm_rw_p.getWidth(),
                               CV_32F, pred_depth_data);
      if (FLAGS_flip_depth_images) {
        cv::flip(depth_img_pred, depth_img_pred, 0);
      }
    } else {
      LOG(FATAL) << "Unknown predicted depth image format "
                 << FLAGS_pred_depth_fmt;
    }

    //     cout << "npy: " << arr.shape[0] << ", " << arr.shape[1] << ", "
    //                     << arr.shape[2] << endl;
    //     cout << "cv img: " << depth_img_pred.size() << endl;

    // Resize all images to the desired size
    if (count < (1 + kSkipFirstFrame)) {
      if (depth_img_pred.size() != image_size) {
        LOG(INFO) << "Resizing predicted depth images to " << image_size;
      }
      if (depth_img_gt.size() != image_size) {
        LOG(INFO) << "Resizing the reference truth depth images to "
                  << image_size;
      }
      if (left_img.size() != image_size) {
        LOG(INFO) << "Resizing the RGB images to " << image_size;
      }
    }

    cv::resize(depth_img_pred, depth_img_pred, image_size);
    cv::resize(depth_img_gt, depth_img_gt, image_size);
    cv::resize(left_img, left_img, image_size);

    cv::Mat obstacle_dist_gt;
    cv::Mat obstacle_img_gt = depth_img_converter.GenerateObstacleImage(
        depth_img_gt, FLAGS_positive_height_obs_thresh,
        FLAGS_negative_height_obs_thresh, &obstacle_dist_gt);
    cv::Mat obstacle_dist_pred;
    cv::Mat obstacle_img_pred = depth_img_converter.GenerateObstacleImage(
        depth_img_pred, FLAGS_positive_height_obs_thresh,
        FLAGS_negative_height_obs_thresh, &obstacle_dist_pred);

    dataset.LabelData(obstacle_img_gt, obstacle_dist_gt, obstacle_img_pred,
                      obstacle_dist_pred, prefix);
    dataset.SaveImages(left_img);

    if (FLAGS_visualization) {
      sensor_msgs::PointCloud2 pointcloud2;
      if (depth_img_converter.GeneratePointcloud(depth_img_gt, 0,
                                                 &pointcloud2)) {
        point_cloud_publisher_gt.publish(pointcloud2);

        sensor_msgs::PointCloud2 pointcloud2_filt =
            depth_img_converter.FilterPointCloudByDistance(pointcloud2, 0,
                                                           FLAGS_max_range);
        point_cloud_publisher_gt_filt_dist.publish(pointcloud2_filt);
      }

      if (depth_img_converter.GeneratePointcloud(depth_img_pred, 0,
                                                 &pointcloud2)) {
        point_cloud_publisher_pred.publish(pointcloud2);

        sensor_msgs::PointCloud2 pointcloud2_filt =
            depth_img_converter.FilterPointCloudByHeight(
                pointcloud2, FLAGS_positive_height_obs_thresh,
                std::numeric_limits<float>::max());
        point_cloud_publisher_pred_filt_height.publish(pointcloud2_filt);
      }
    }

    //     count++;
    //     if (count > 30) {
    //       break;
    //     }

    //       // Visualize and verify the depth image
    //     double min;
    //     double max;
    //     cv::minMaxIdx(depth_img_gt, &min, &max);
    //     cv::Mat adjMap;
    //     cv::convertScaleAbs(depth_img_gt, adjMap, 255.0 / max);
    //     cv::imshow("window", adjMap);

    // TODO: Remove these debugging visualizations
    // cv::Mat in_range_mask;
    // cv::threshold(depth_img_gt, in_range_mask, 0.01, 1, cv::THRESH_BINARY);
    // in_range_mask.convertTo(in_range_mask, CV_8U);
    // cv::Mat mask_vis;
    // cv::convertScaleAbs(in_range_mask, mask_vis, 255);
    // cv::imshow("GT Depth in range over depth", mask_vis);

    // cv::Mat in_range_mask_pred;
    // cv::threshold(depth_img_pred, in_range_mask_pred, 0.01, 1,
    // cv::THRESH_BINARY); in_range_mask_pred.convertTo(in_range_mask_pred,
    // CV_8U); cv::Mat mask_vis_pred; cv::convertScaleAbs(in_range_mask_pred,
    // mask_vis_pred, 255); cv::imshow("Predicted Depth in range over depth",
    // mask_vis_pred);

    count++;
    if (FLAGS_visualization && FLAGS_debug) {
      cout << "img num: " << i << endl;
      cv::imshow("window", left_img);
      cv::waitKey(0);
    }
  }

  LOG(INFO) << "Saving the dataset ...";
  dataset.SaveToFile();

  return 0;
}
