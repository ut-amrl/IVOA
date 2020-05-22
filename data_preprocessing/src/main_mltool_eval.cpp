// Copyright 2020 srabiee@cs.utexas.edu, kvsikand@cs.utexas.edu
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

#include <ros/ros.h>
#include <ros/console.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <iomanip>
#include <jsoncpp/json/json.h>
#include "yaml-cpp/yaml.h"

#include "io_access.h"
#include "depth2pointcloud.h"
#include "evaluator.h"
#include "cnpy.h"

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

#define DEBUG false

using cv::Mat;
using namespace std;
using namespace IVOA;
using std::vector;
using cv::Point;


// Command line flags flag
DEFINE_int32(session_num, 0, "Session number to identify the generated "
                             "portion of the dataset.");
DEFINE_string(source_dir, "", "Path to the base directory of the source " 
                              "dataset.");
DEFINE_string(cam_extrinsics_path, "", "Path to the file containing the "
                                       "left camera calibration file.");
DEFINE_string(trajectory_path, "", "Path to the trajectory file to use.");
DEFINE_string(output_dir, "", "Path to save the generated results. ");
DEFINE_double(max_range, 40.0, "Max range to consider for depth prediction evaluation");
DECLARE_bool(help);
DECLARE_bool(helpshort);


// Parameters
const bool kVisualization = true;

// Objects that are further than max_range_ away from the agent will not be
// considered for obstacle avoidance
const float kMinObstacleHeight = 0.3; // meters
const float kMaxObstacleHeight = 2.0; // meters

// If the predicted distance to obstacle is off from the ground truth by
// more than max(kDistanceErrThresh, kRelativeErrThresh * TrueDistance), it 
// will be labeled as either FP (if pred_dist < gt_dist) or FN (if pred_dist > 
// gt_dist)
const float kDistanceErrThresh = 1.0; // meters
const float kRelativeDistanceErrThresh = 0.0; // ratio in [0, 1]

// TODO: Should kMarginWidth be a command line argument?
// Remove depth predictions in the margins of the depth image
const int kMarginWidth = 50;

// TODO: Load camera intrinsics from file as well, so that the scaled down 
// version of depth predictions could be supported


// Parameters of the virtual 2D laser scan
const float kMinRange = 0.1; // meters
const float kAngleIncrementLaser = 0.5 * M_PI / 180.0;

// The minimum length below which we don't visualize error tracks
const int kErrorTrackMinLength = 1;
// The lenght to which we normalize track length opacity
const int kErrorTrackMaxLength = 25;

const int kVisualizationErrorSize = 4;

const cv::Size kImageSize(960, 600);

// Checks if all required command line arguments have been set
void CheckCommandLineArgs(char** argv) {
  vector<string> required_args = {"session_num",
                                  "source_dir",
                                  "cam_extrinsics_path",
                                  "output_dir",
                                  "trajectory_path"};
  
  for (const string& arg_name:required_args) {
    bool flag_not_set =   
          gflags::GetCommandLineFlagInfoOrDie(arg_name.c_str()).is_default;
    if (flag_not_set) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "main_mltool_eval");
      LOG(FATAL) << arg_name <<  " was not set." << endl;
    }
  }
}

void GetTrajectoryPoses(const std::string &path,
                         std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternion<float>>> *trajectory) {
  std::ifstream trajFile(path);
  std::string line;

  while(std::getline(trajFile, line)) {
    std::stringstream ss(line);

    string timestamp;
    std::getline(ss, timestamp, ',');

    string skip;
    // Ignore the next 3 elements (speed, gear, rpm)
    std::getline(ss, skip, ',');
    std::getline(ss, skip, ',');
    std::getline(ss, skip, ',');

    string pos_x;
    std::getline(ss, pos_x, ',');
    string pos_y;
    std::getline(ss, pos_y, ',');
    string pos_z;
    std::getline(ss, pos_z, ',');
    Eigen::Vector3f pose(stof(pos_x), stof(pos_y), stof(pos_z));

    string orientation_x;
    std::getline(ss, orientation_x, ',');
    string orientation_y;
    std::getline(ss, orientation_y, ',');
    string orientation_z;
    std::getline(ss, orientation_z, ',');
    string orientation_w;
    std::getline(ss, orientation_w, ',');
    Eigen::Quaternion<float> orientation(stof(orientation_x), stof(orientation_y), stof(orientation_z), stof(orientation_w));
    trajectory->push_back(std::make_pair(pose, orientation));
  }
}


int main(int argc, char **argv) {
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;   // INFO level logging.
  FLAGS_colorlogtostderr = 1;  // Colored logging.
  FLAGS_logtostderr = true;    // Don't log to disk
 
  
  string usage("This program detects and keeps track of failures in "
          "obstacle avoidance for a traveresed trajectory by the robot "
          "given the reference depth estimates as well as the predicted "
          "depth images by an ml model: \n");

  usage += string(argv[0]) + " <argument1> <argument2> ...";
  gflags::SetUsageMessage(usage);

  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_help) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "main_mltool_eval");
    return 0;
  }
  CheckCommandLineArgs(argv);
  
  ros::init(argc, argv, "mltool_evaluation");
  ros::NodeHandle nh;
 
  
  ros::Publisher point_cloud_publisher_gt =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/gt_pointcloud", 1);
  ros::Publisher point_cloud_publisher_pred =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/pred_pointcloud", 1);
  ros::Publisher point_cloud_publisher_gt_filt =
      nh.advertise<sensor_msgs::PointCloud2>(
          "/ivoa/gt_pointcloud_filtered", 1);
  ros::Publisher point_cloud_publisher_pred_filt =
      nh.advertise<sensor_msgs::PointCloud2>(
          "/ivoa/pred_pointcloud_filtered", 1);
  ros::Publisher point_cloud_publisher_err =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/err_pointcloud", 1);
  ros::Publisher laserscan_publisher_gt =
      nh.advertise<sensor_msgs::LaserScan>("/ivoa/gt_laserscan", 1);
  ros::Publisher laserscan_publisher_pred =
      nh.advertise<sensor_msgs::LaserScan>("/ivoa/pred_laserscan", 1);
  ros::Publisher fp_laserscan_publisher =
      nh.advertise<sensor_msgs::LaserScan>("/ivoa/fp_laserscan", 1);
  ros::Publisher fn_laserscan_publisher =
      nh.advertise<sensor_msgs::LaserScan>("/ivoa/fn_laserscan", 1);
  ros::Publisher point_cloud_publisher_gt_global =
      nh.advertise<sensor_msgs::PointCloud2>("/ivoa/gt_global", 1);

  ros::Publisher error_track_publisher =
      nh.advertise<visualization_msgs::MarkerArray>("/ivoa/error_tracks", 1);

  ros::Publisher pose_publisher =
      nh.advertise<geometry_msgs::PoseStamped>("/ivoa/pose", 1);
  
  ros::Publisher trajectory_publisher =
      nh.advertise<geometry_msgs::PoseArray>("/ivoa/trajectory", 1);
  geometry_msgs::PoseArray pa;

  // Using only one instance of Depth2Pointcloud since the depth and left RGB
  // camera share the same extrinsics and intrinsics in our AirSim setup 
  Depth2Pointcloud depth_img_converter;
  depth_img_converter.LoadCameraCalibration(FLAGS_cam_extrinsics_path);
  
  Evaluator evaluator(kDistanceErrThresh,
                      kRelativeDistanceErrThresh,
                      kVisualization);
  evaluator.LoadCameraCalibration(FLAGS_cam_extrinsics_path);
 

  string gt_depth_dir = FLAGS_source_dir + "/img_depth/";
  string pred_depth_dir = FLAGS_source_dir + "/img_left/";
  string left_img_dir = FLAGS_source_dir + "/img_left/";
 
  // Read the prefix of all avaiable datapoints
  vector<int> filename_prefixes;
  GetFileNamePrefixes(gt_depth_dir,
                      &filename_prefixes);
  std::sort(filename_prefixes.begin(), filename_prefixes.end());
  
 
  // For each point in the trajectory, load the pose and orientation
  // They should be in the order of the filenames
  vector<std::pair<Eigen::Vector3f, Eigen::Quaternion<float>>> trajectory;
  GetTrajectoryPoses(FLAGS_trajectory_path,
                      &trajectory);

  // If this isn't true, we likely aren't using the correct pruned trajectory
  CHECK_EQ(trajectory.size(), filename_prefixes.size());
  
  
  // Transform from NED to the ROS standard 
  Eigen::Matrix4f T_NED2ROS = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_ROS2NED = Eigen::Matrix4f::Identity();
  T_NED2ROS << 1.0000000,  0.0000000,  0.0000000, 0.0000000,
                0.0000000, -1.0000000, -0.0000000, 0.0000000,
                0.0000000,  0.0000000, -1.0000000, 0.0000000,
                0.0000000,  0.0000000,  0.0000000, 1.0000000;
  T_ROS2NED = T_NED2ROS;

  
  int count = 0;
  for (int idx = 0; idx < filename_prefixes.size(); idx++) {
    const int& i = filename_prefixes[idx];
    
    stringstream ss;
    ss << setfill('0') << setw(10) << i;
    string prefix = ss.str();
    string gt_depth_path = gt_depth_dir + prefix + ".pfm";
    string pred_depth_path = pred_depth_dir + prefix + "_disp.npy";
    string left_img_path = left_img_dir + prefix + ".png";

    // Read the Ground truth depth image
    PFM pfm_rw;
    float * depth_data = 
                  pfm_rw.read_pfm<float>(gt_depth_path);
    cv::Mat depth_img_gt = cv::Mat(pfm_rw.getHeight(), 
                                pfm_rw.getWidth(), 
                                CV_32F, 
                                depth_data);
    cv::flip(depth_img_gt, depth_img_gt, 0);
    

    // Read the predicted depth image
    cnpy::NpyArray arr = cnpy::npy_load(pred_depth_path);
    float* loaded_data = arr.data<float>();
    cv::Mat depth_img_pred = cv::Mat(arr.shape[0],
                                     arr.shape[1],
                                     CV_32F,
                                     loaded_data);
  
    // TODO: Revisit how to resize the depth prediction. Maybe scale down the 
    // ground truth depth to match the predicted depth image.
    
    // Resize and linearly interpolate the predicted depth image so that it 
    // is the same size as the ground truth depth image
    cv::resize(depth_img_pred,depth_img_pred,depth_img_gt.size());
    
    // Ground truth and predicted point clouds
    sensor_msgs::PointCloud2 pt_cloud_gt;
    sensor_msgs::PointCloud2 pt_cloud_pred;
    if (!depth_img_converter.GeneratePointcloud(depth_img_gt, 
                                                kMarginWidth,
                                                &pt_cloud_gt) ||
      !depth_img_converter.GeneratePointcloud(depth_img_pred,
                                              kMarginWidth, 
                                              &pt_cloud_pred)){
      LOG(FATAL) << "Point cloud generation failed. "
                 << "No camera calibration was available.";
    }
    
    // Publish the original point clouds
    if (kVisualization) {
      point_cloud_publisher_gt.publish(pt_cloud_gt);
      point_cloud_publisher_pred.publish(pt_cloud_pred);
    }
    
    std::pair<Eigen::Vector3f, Eigen::Quaternion<float>> pose = trajectory[idx];
    // std::cout << "POSE: " << pose.first.transpose() << std::endl;
    // std::cout << "ORIENTATION: " << pose.second.transpose() << std::endl;
    Eigen::Matrix4f T_baseNED2mapNED;
    T_baseNED2mapNED.setIdentity();
    T_baseNED2mapNED.block<3,3>(0,0) = pose.second.toRotationMatrix();
    T_baseNED2mapNED.block<3,1>(0,3) = pose.first;
    
    // Transformation from the base link in the standard ROS coordinate to 
    // the map reference frame which is also in the standard ROS coordinate
    // (forward: x, left: y, up: z)
    Eigen::Matrix4f T_base2map;
    T_base2map.setIdentity();
    T_base2map = T_NED2ROS * T_baseNED2mapNED * T_ROS2NED; 

    if (kVisualization) {
      sensor_msgs::PointCloud2 pt_cloud_gt_global;
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_gt(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_gt_global(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(pt_cloud_gt, *pcl_cloud_gt);
      pcl::transformPointCloud(*pcl_cloud_gt, *pcl_cloud_gt_global, T_base2map);
      pcl::toROSMsg(*pcl_cloud_gt_global, pt_cloud_gt_global);
      
      pt_cloud_gt_global.header.frame_id = "map";
      point_cloud_publisher_gt_global.publish(pt_cloud_gt_global);
    }
    
    // Filter point clouds by height and range
    pt_cloud_gt = depth_img_converter.FilterPointCloudByHeightAndDistance(
                                                            pt_cloud_gt,
                                                            kMinObstacleHeight,
                                                            kMaxObstacleHeight,
                                                            0,
                                                            FLAGS_max_range);
    
    pt_cloud_pred = depth_img_converter.FilterPointCloudByHeightAndDistance(
                                                            pt_cloud_pred,
                                                            kMinObstacleHeight,
                                                            kMaxObstacleHeight,
                                                            0,
                                                            FLAGS_max_range);

    ProjectedPtCloud proj_ptcloud_pred;
    ProjectedPtCloud proj_ptcloud_gt;
    
    depth_img_converter.GenerateProjectedPtCloud(depth_img_pred,
                                            kMarginWidth,     
                                            kAngleIncrementLaser,  
                                            kMinRange,        
                                            FLAGS_max_range,
                                            kMinObstacleHeight,
                                            kMaxObstacleHeight,
                                            &proj_ptcloud_pred);
   
              
    depth_img_converter.GenerateProjectedPtCloud(depth_img_gt,
                                            kMarginWidth,     
                                            kAngleIncrementLaser,  
                                            kMinRange,        
                                            FLAGS_max_range,
                                            kMinObstacleHeight,
                                            kMaxObstacleHeight,
                                            &proj_ptcloud_gt);
    
    // std::cout << "Transformation: \n" << T_base2map << std::endl;

    unsigned int errors_idx = evaluator.EvaluatePredictions(proj_ptcloud_pred,
                                  proj_ptcloud_gt,
                                  T_base2map,
                                  static_cast<long unsigned int>(i),
                                  depth_img_gt);  
    
    std::vector<Evaluator::Error> errors = evaluator.GetErrors()[errors_idx];
    // std::cout << "Example error location: " << errors[0].loc_map.transpose() << std::endl;

    // Publish the filtered point clouds
    if (kVisualization) {
      point_cloud_publisher_gt_filt.publish(pt_cloud_gt);
      point_cloud_publisher_pred_filt.publish(pt_cloud_pred);
      point_cloud_publisher_err.publish(evaluator.GetErrorsPointCloud());
      
      sensor_msgs::LaserScan laserscan_gt = 
            depth_img_converter.ProjectedPtCloud_to_LaserScan(proj_ptcloud_gt);
      sensor_msgs::LaserScan laserscan_pred = 
           depth_img_converter.ProjectedPtCloud_to_LaserScan(proj_ptcloud_pred);
      laserscan_publisher_gt.publish(laserscan_gt);
      laserscan_publisher_pred.publish(laserscan_pred);
      
      fp_laserscan_publisher.publish(evaluator.GetFalsePositivesScan());
      fn_laserscan_publisher.publish(evaluator.GetFalseNegativesScan());
      
      // Convert from NED to ROS standard
      Eigen::Matrix3f rot_base2map;
      rot_base2map = T_base2map.topLeftCorner(3,3);
      Eigen::Quaternion<float> quat_base2map(rot_base2map);

      geometry_msgs::PoseStamped pose_msg;

      pose_msg.header.stamp = ros::Time::now();
      pose_msg.header.frame_id = "base_link";
      pose_msg.pose.position.x = T_base2map(0,3);
      pose_msg.pose.position.y = T_base2map(1,3);
      pose_msg.pose.position.z = T_base2map(2,3);

      pose_msg.pose.orientation.x = quat_base2map.x();
      pose_msg.pose.orientation.y = quat_base2map.y();
      pose_msg.pose.orientation.z = quat_base2map.z();
      pose_msg.pose.orientation.w = quat_base2map.w();

      pa.poses.push_back(pose_msg.pose);
      pose_publisher.publish(pose_msg);
    }

    if (kVisualization) {
      // Read the left cam image
      Mat left_img_annotated = cv::imread(left_img_path,CV_LOAD_IMAGE_UNCHANGED);
      
      for(Evaluator::Error e : errors) {
        cv::Scalar green = cv::Scalar(0, 255 , 0);
        cv::Scalar red = cv::Scalar(0, 0 , 255);
        cv::Scalar color;
        if (e.error_type == Evaluator::PredictionLabel::FP) {
          color = red;
        } else {
          color = green;
        }

        cv::circle(left_img_annotated, cv::Point(e.pixel_coord.x(), e.pixel_coord.y()), 2, color, -1, 8, 0);
      }
      #if DEBUG
      cout << "img num: " << i << endl;
      cv::imshow("window", left_img_annotated);
      cv::waitKey(0);
      #endif
    }

    count++;
  }

  std::vector<unsigned long int>prediction_label_counts;
  prediction_label_counts = evaluator.GetStatistics();
  printf("Writing statistics information to file...\n");
  ofstream stats_file;
  stats_file.open(FLAGS_output_dir + "/" + "prediction_label_statistics.txt");

  stats_file << "False Positives: " << prediction_label_counts[Evaluator::PredictionLabel::FP] << std::endl;
  stats_file << "False Negatives: " << prediction_label_counts[Evaluator::PredictionLabel::FN] << std::endl;
  stats_file << "True Positives: " << prediction_label_counts[Evaluator::PredictionLabel::TP] << std::endl;
  stats_file << "True Negatives: " << prediction_label_counts[Evaluator::PredictionLabel::TN] << std::endl;

  stats_file << "Total Examples: " << std::accumulate(prediction_label_counts.begin(), prediction_label_counts.end(), 0) << std::endl;
  stats_file.close();

  if (kVisualization) {
    printf("Publishing error track markers...\n");
    visualization_msgs::MarkerArray ma;
    int id = 0;
    for(Evaluator::ErrorTrack track : evaluator.GetErrorTracks()) {
      visualization_msgs::Marker m;
      if (track.loc_map_history.size() < kErrorTrackMinLength) {
        continue;
      }

      m.pose.position.x = track.loc_map_history[0].second[0];
      m.pose.position.y = track.loc_map_history[0].second[1];
      m.pose.position.z = track.loc_map_history[0].second[2];
      m.pose.orientation.x = 0.0;
      m.pose.orientation.y = 0.0;
      m.pose.orientation.z = 0.0;
      m.pose.orientation.w = 1.0;

      m.scale.x = 1.0;
      m.scale.y = 1.0;
      m.scale.z = 1.0;

      m.ns = "error_tracks";
      m.id = id++;
      m.header.frame_id = "map";

      // float alpha = std::min(1.0f, (float)(track.loc_map_history.size() - kErrorTrackMinLength) / (kErrorTrackMaxLength - kErrorTrackMinLength));
      if (track.error_type == Evaluator::PredictionLabel::FP) {
        m.color.r = 1.0f;
        m.color.g = 0.0f;
        m.color.b = 0.0f;
      } else {
        m.color.r = 0.0f;
        m.color.g = 0.0f;
        m.color.b = 1.0f;
      }
      m.color.a = 1.0f;
      ma.markers.push_back(m);
    }

    error_track_publisher.publish(ma);

    pa.header.frame_id = "map";
    trajectory_publisher.publish(pa);

    printf("Writing histogram information to file...\n");
    
    Evaluator::ErrorHistogram track_hist = evaluator.getErrorTrackSizeHistogram();
    ofstream track_file;
    track_file.open(FLAGS_output_dir + "/" + "error_track_size_histogram.csv");
    for(auto bucket : track_hist.buckets) {
      track_file << bucket.lower << ", " << bucket.upper << ", " << bucket.count << std::endl;
    }
    track_file.close();
    
    Evaluator::ErrorHistogram histogram = evaluator.getAbsoluteDistanceErrorHistogram();
    ofstream hist_file;
    hist_file.open(FLAGS_output_dir + "/" + "dist_error_histogram.csv");
    for(auto bucket : histogram.buckets) {
      hist_file << bucket.lower << ", " << bucket.upper << ", " << bucket.count << std::endl;
    }
    hist_file.close();

    ofstream hist_window_file;
    hist_window_file.open(FLAGS_output_dir + "/" + "dist_error_histogram_windows.csv");
    for(auto window : histogram.windows) {
      hist_window_file << window.pct << ", " << window.pos_bound << ", " << window.neg_bound << std::endl;
    }
    hist_window_file.close();

    Evaluator::ErrorHistogram rel_histogram = evaluator.getRelativeDistanceErrorHistogram();
    ofstream rel_hist_file;
    rel_hist_file.open(FLAGS_output_dir + "/" + "rel_dist_error_histogram.csv");
    for(auto bucket : rel_histogram.buckets) {
      rel_hist_file << bucket.lower << ", " << bucket.upper << ", " << bucket.count << std::endl;
    }
    rel_hist_file.close();

    ofstream rel_hist_window_file;
    rel_hist_window_file.open(FLAGS_output_dir + "/" + "rel_dist_error_histogram_windows.csv");
    for(auto window : rel_histogram.windows) {
      rel_hist_window_file << window.pct << ", " << window.pos_bound << ", " << window.neg_bound << std::endl;
    }
    rel_hist_window_file.close();
  }
  return 0;
}

