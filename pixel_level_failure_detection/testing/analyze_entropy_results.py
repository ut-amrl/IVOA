#!/bin/python

# ========================================================================
# Copyright 2022 srabiee@cs.utexas.edu
# Department of Computer Science,
# University of Texas at Austin


# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License Version 3,
# as published by the Free Software Foundation.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# Version 3 in the file COPYING that came with this distribution.
# If not, see <http://www.gnu.org/licenses/>.
# ========================================================================

"""
This script loads the saved entropy of predictions and visualizes the results.
"""

import argparse, os
from email.policy import default
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def main(args):
  
  entropy_vals_all_list = []
  assert len(args.src_directories) == len(args.dataset_names), "Number of src directories and dataset names must match"
  
  dataset_num = len(args.src_directories)
  max_range = -np.Inf
  min_range = np.Inf
  
  for i in range(dataset_num):
    src_dir = args.src_directories[i]
  
    # Go through all the files in the directory and load them into a np.array
    entropy_vals = np.array([], dtype=np.float_)
    
    # Get the list of files in the directory
    files = os.listdir(src_dir)
    print("Loading files from {}".format(src_dir))
    for file in tqdm(files):
      file_path = os.path.join(src_dir, file)
      entropy = np.load(file_path)
      entropy_vals = np.concatenate((entropy_vals, entropy))

    print("Total datapoints loaded: {}".format(entropy_vals.size))
    
    
    # Set nan values to 0 (entropy of p_failure=0 is 0)
    entropy_vals[np.isnan(entropy_vals)] = 0
    
    print("entropy_vals.max(): {}".format(np.max(entropy_vals)))
    print("entropy_vals.min(): {}".format(np.min(entropy_vals)))
    max_range = max(max_range, np.max(entropy_vals))
    min_range = min(min_range, np.min(entropy_vals))
    
    entropy_vals_all_list += [entropy_vals]
    
    
  # TODO: Beautify the plot
  output_file = os.path.join(args.output_dir, 'entropy_cdf.pdf')
  drawing_styles = ['b-', 'r-', 'g-', 'y-', 'c-', 'm-', 'k-']
  with PdfPages(output_file) as pdf_file:
    for i in range(dataset_num):
      entropy_vals = entropy_vals_all_list[i]
      dataset_name = args.dataset_names[i]
      
      hist, bin_edges = np.histogram(entropy_vals, bins=100, range=(min_range, max_range)) 
      
      sum = np.sum(hist)
      pdf = hist / sum
      cdf = np.cumsum(pdf)
      
      plt.plot(bin_edges[1:], cdf, drawing_styles[i], label=dataset_name)
      
    plt.legend()
    plt.xlabel("Entropy")
    plt.ylabel("CDF")
    pdf_file.savefig()  # saves the current figure into a pdf page
    plt.close()
    
  # TODO: Maybe bring back the plot of pdf?
    
  # # Generate the histogram of the entropy values
  # bin_num = 100
  # # bin_num = 10
  # hist, bin_edges = np.histogram(entropy_vals, bins=bin_num)
  
  # # Generate PDF and CDF
  # sum = np.sum(hist)
  # pdf = hist / sum
  # cdf = np.cumsum(pdf)
  

  
  # output_file = os.path.join(args.output_dir, 'entropy_pdf_cdf.pdf')
  # with PdfPages(output_file) as pdf_file:
  #   # Plot the PDF and CDF
  #   plt.plot(bin_edges[1:], pdf, 'r-', label='PDF')
  #   plt.plot(bin_edges[1:], cdf, 'b-', label='CDF')
  #   plt.legend(loc='upper right')
  #   pdf_file.savefig()  # saves the current figure into a pdf page
  #   plt.close()
  
  
  
  

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description='This script loads the saved entropy of predictions and visualizes the results.')
  
  parser.add_argument(
    "--src_directories",
    nargs="+",
    default=[
      # "/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_city_wb_ensemble/evaluation_multi_class_test_per_image//ivoa_pix_ensemble_epoch_14_e012_mobilenet_c1deepsup/entropy", 
      # "/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_africa_wb_ensemble/evaluation_multi_class_test_per_image/ivoa_pix_ensemble_epoch_14_e012_mobilenet_c1deepsup/entropy"], # IPr Ensemble - Africa dataset difficult
      
      # "/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_city_wb_ensemble/evaluation_multi_class_test_per_image//ivoa_pix_ensemble_epoch_14_e012_mobilenet_c1deepsup/entropy", 
      # "/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_africa_wb_ensemble/evaluation_multi_class_test_per_image_africa_v1.4_02_ood_tmp_var_thresh_0.0005/ivoa_pix_ensemble_epoch_14_e012_mobilenet_c1deepsup/entropy"], # IPr Ensemble - Africa dataset easy (v1.4 traj and v1.4 camera)
      
      # "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/cityenv_wb_manual_calib/entropy/", 
      # "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/africa_wb_tmp_v1.4/entropy/"], #  Ensemble - Africa dataset easy (v1.4 traj and v1.4 camera)
      
      "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/cityenv_wb_manual_calib/entropy/", 
      "/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/africa_wb/entropy/"], #  Ensemble - Africa dataset difficult
    required=False,
    help="The path to src directory including the npy entropy files for different datasets.")
  parser.add_argument(
    "--dataset_names",
    nargs="+",
    default=["ID", "OOD"],
    required=False,
    help="Names of the datasets corresponding to the src_directories.")
  
  parser.add_argument(
                      "--src_dir",
                      # default="/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_africa_wb_ensemble/evaluation_multi_class_test_per_image/ivoa_pix_ensemble_epoch_14_e012_mobilenet_c1deepsup/entropy", # ipr ensemble africa 
                      # default="/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_city_wb_ensemble/evaluation_multi_class_test_per_image//ivoa_pix_ensemble_epoch_14_e012_mobilenet_c1deepsup/entropy", # ipr ensemble city
                      default="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/africa_wb/entropy", # ensemble africa
                      # default="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/cityenv_wb_manual_calib/entropy/", # ensemble city
                      help="path to src directory including the npy entropy files.", 
                      type=str,
                      required=False)
  
  parser.add_argument(
                      "--output_dir",
                      # default="/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_africa_wb_ensemble/evaluation_multi_class_test_per_image_africa_v1.4_02_ood_tmp_var_thresh_0.0005/", # ipr ensemble africa  dataset easy (v1.4 traj and v1.4 camera)
                      
                      # default="/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_africa_wb_ensemble/evaluation_multi_class_test_per_image/", # ipr ensemble africa
                      
                      # default="/robodata/srabiee/scratch/results/IVOA/evaluation/GANET_PIX/ganet_deep_airsim_01_mcdropoutOff_r0.0_epoch_14_p36_rg30_errTh1.0_0.2_NoGP_ds_pix_city_wb_ensemble/evaluation_multi_class_test_per_image/", # ipr ensemble city
                      
                      default="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/africa_wb/", # ensemble africa
                      
                      # default="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/africa_wb_tmp_v1.4/", # ensemble africa dataset easy (v1.4 traj and v1.4 camera)
                      
                      # default="/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012/cityenv_wb_manual_calib/", # ensemble city
                      help="Dir to save the result figures to.", 
                      type=str,
                      required=False)
  
  args = parser.parse_args()
  
  main(args)