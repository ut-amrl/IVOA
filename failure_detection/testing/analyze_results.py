# Copyright 2019 srabiee@cs.umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst


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

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import copy
import argparse
from skimage import io, transform
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from math import floor
from collections import OrderedDict
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from matplotlib.backends.backend_pdf import PdfPages
from test_obstacle_detection_model import plot_confusion_matrix

def load_result_files(file_paths):
    all_results = []
    for path in file_paths:
        with open(path, 'r') as file:
            all_results = all_results + [json.load(file)]
    return all_results        

def merge_results(result_dics):
    total_ground_truth = []
    total_predictions = []
    total_class_prob = {}
    total_class_uncertainty = {}
    classes = []
   
    classes = result_dics[0]["classes"]
    has_uncertainty = bool(result_dics[0]["class_uncertainty"])
    
    for curr_class in classes:
          total_class_prob[curr_class] = []
          
    if has_uncertainty:
        for curr_class in classes:
            total_class_uncertainty[curr_class] = []
    else:
        print("Loaded data does not include uncertainty information.")
    
    for curr_result in result_dics:
          total_ground_truth = (total_ground_truth + 
                                curr_result["ground_truth"])
          total_predictions = (total_predictions + 
                                curr_result["predictions"])
          for curr_class in classes:
              total_class_prob[curr_class] = (total_class_prob[curr_class] +
                            curr_result["class_prob"][curr_class])
          
          if has_uncertainty:
              for curr_class in classes:
                  total_class_uncertainty[curr_class] = (
                                total_class_uncertainty[curr_class] +
                                curr_result["class_uncertainty"][curr_class])
          
    
    merged_result = {"classes": classes,
                    "ground_truth": total_ground_truth,
                    "predictions": total_predictions,
                    "class_prob": total_class_prob,
                    "class_uncertainty": total_class_uncertainty}
    return merged_result
  
def save_results(target_dir, file_name, result_dic):
    results_json = json.dumps(result_dic, indent=2)
    file = open(target_dir +'/'+file_name+'_data.json', 'w');
    file.write(results_json)
    file.close()
    
# Converts the dictionary format of the results to numpy arrays    
def convert_results_to_np(results_dict):
    data_num = len(results_dict["ground_truth"])
    classes = results_dict["classes"]
    class_num = len(classes)
    class_prob = np.zeros((data_num, class_num))
    class_uncertainty = np.zeros((data_num, class_num))
    
    for i in range(class_num):
        class_prob[:,i] = results_dict["class_prob"][classes[i]]
        class_uncertainty[:,i] = results_dict["class_uncertainty"][classes[i]]
    
    results_np = {"classes": classes,
                  "ground_truth": np.array(results_dict["ground_truth"]),
                  "predictions": np.array(results_dict["predictions"]),
                  "class_prob": class_prob,
                  "class_uncertainty": class_uncertainty}
    return results_np
  
# Returns the indices of the correct and wrong predictions
def check_predictions(results_np):
    correctness = results_np["predictions"] == results_np["ground_truth"]
    
    correct_ind = np.asarray(np.where(correctness))
    wrong_ind = np.asarray(np.where(np.logical_not(correctness)))
    
    return correct_ind, wrong_ind
  
def get_pred_class_score(results_np):
    print(results_np["class_prob"].shape)
    print(results_np["predictions"].shape)
    return results_np["class_prob"][range(results_np["predictions"].size), 
                                    results_np["predictions"]]
  
def get_pred_class_var(results_np):
    return results_np["class_uncertainty"][
            range(results_np["predictions"].size), results_np["predictions"]]
 
# Returns a mask that selects the confident predictions based on the given 
# uncertainty/variance threshold
def filter_unconfident(results_np, var_thresh):
    pred_class_var = get_pred_class_var(results_np)
    confident_mask = pred_class_var <= var_thresh
    
    return confident_mask
  
# Returns a mask that selects the confident predictions based on the given 
# prediction score threshold. Note that this method assumes predictions where
# the predicted class has a score of higher than class_score_thresh, are 
# of high accuracy
def filter_on_class_score(results_np, class_score_thresh):
    pred_class_score = get_pred_class_score(results_np)
    confident_mask = pred_class_score >= class_score_thresh
    
    return confident_mask
  
# Returns the number of samples that exist from each class in the dataset 
# A filtering mask is also provided to the function so that counting only
# happens on the filtered part of the dataset
def count_data_labels(results_np, filter_mask):
    class_num = len(results_np["classes"])
    class_count = np.zeros((class_num))
    for i in range(class_num):
        class_count[i] = np.sum(results_np["ground_truth"][filter_mask] == i) 
    return class_count
  
def draw_acc_vs_uncertainty(results_np, uncertainty_thresh_vals,
                            save_dir, result_file_name):
    data_num = len(results_np["predictions"])
    class_count_orig = count_data_labels(results_np, 
                                         np.ones(data_num, dtype=bool))
    print(class_count_orig)
    
    class_num = len(results_np["classes"])
    classes = results_np["classes"]
    acc_vals = np.zeros(len(uncertainty_thresh_vals))
    f1_micro_vals = np.zeros(len(uncertainty_thresh_vals))
    f1_macro_vals = np.zeros(len(uncertainty_thresh_vals))
    class_count = np.zeros((len(uncertainty_thresh_vals), class_num))
    
    # Calculate different evaluation metrics for the whole multi-class 
    # classification task: accuracy, f1_micro, f1_macro
    for i in range(len(uncertainty_thresh_vals)):
        filter_mask = filter_unconfident(results_np, uncertainty_thresh_vals[i])
        #print(np.sum(filter_mask))
        acc_vals[i] = accuracy_score(results_np["ground_truth"][filter_mask], 
                                      results_np["predictions"][filter_mask],
                                      normalize = True)
        f1_micro_vals[i] = f1_score(results_np["ground_truth"][filter_mask], 
                                    results_np["predictions"][filter_mask],
                                    average='micro')
        f1_macro_vals[i] = f1_score(results_np["ground_truth"][filter_mask], 
                                    results_np["predictions"][filter_mask],
                                    average='macro')
        class_count[i, :] = count_data_labels(results_np, filter_mask)
     
    acc_vals_percent = np.multiply(acc_vals, 100.0)
    all_class_count = np.sum(class_count, 1)
    all_class_count_nomalized = np.divide(all_class_count, data_num)
    all_class_count_percent = np.multiply(all_class_count_nomalized, 100.0)
    
    eval_metrics = {"Accuracy": {"scores": acc_vals_percent,
                                 "label": "Accuracy %"},
                    "F1-micro": {"scores": f1_micro_vals,
                                 "label": "F1-micro"},
                    "F1-macro": {"scores": f1_macro_vals,
                                 "label": "F1-macro"}}
        
    for metric, val in eval_metrics.items():
        # Plots accuracy and retained data percentage on the same figure
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Uncertainty Threshold')
        ax1.set_ylabel(val["label"], color=color)
        ax1.plot(uncertainty_thresh_vals, val["scores"], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Retained Data %', color=color) 
        ax2.plot(uncertainty_thresh_vals, all_class_count_percent, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + '/' + result_file_name + '_' + metric + '.png')
        
    # Plot retained data percentage vs. uncertainty thresh for each class
    fig = plt.figure()
    for i in range(class_num):
        plt.plot(uncertainty_thresh_vals, 
            np.divide(class_count[:,i], class_count_orig[i]), 
                      label=classes[i])
        plt.xlabel("Uncertainty Threshold")
        plt.ylabel("Retained Data %")
        plt.legend()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + '/' + result_file_name + '_data_retension.png')
          
    #plt.show();

# Calculates and draws the confusion matrix for a subset of predictions
# Only those with certainty higher than a threshold
def draw_confusion_mat_confident_res(results_np, uncertainty_thresh_val,
                                     file_name, file_path):
    mask = filter_unconfident(results_np, uncertainty_thresh_val)
    cnf_matrix = confusion_matrix(results_np["ground_truth"][mask],
                                  results_np["predictions"][mask])
    if np.any(np.isnan(cnf_matrix)):
        print("Confusion matrix has NAN values for uncertainty threshold of ",
              uncertainty_thresh_val, ". Skipping the plotting!")
        return 1
    
    title = "Uncertainty thresh: {0:.3f}".format(uncertainty_thresh_val)

    plot_confusion_matrix(cnf_matrix, results_np["classes"], 
              file_name, 
              file_path,
              normalize=True,
              title= title,
              cmap=plt.cm.Blues)
    
# Calculates and draws the confusion matrix for a subset of predictions
# that is defined by the given mask
def draw_confusion_mat_with_mask(results_np, mask,
                                file_name, file_path, title):
    cnf_matrix = confusion_matrix(results_np["ground_truth"][mask],
                                  results_np["predictions"][mask])
    if np.any(np.isnan(cnf_matrix)):
        print("Confusion matrix has NAN values! Skipping the plotting!")
        return 1
   
    plot_confusion_matrix(cnf_matrix, results_np["classes"], 
              file_name, 
              file_path,
              normalize=True,
              title= title,
              cmap=plt.cm.Blues)
    
        
if __name__ == "__main__":
    source_dir = ("/hdd/results/introspective_failure_detection/"
                  "alex_multi_7_color_noMedFilt")
    
    
    
    target_dir = source_dir
    graph_save_dir = target_dir + '/plots/'
    conf_mat_save_dir = target_dir + '/confusion_mat/'
    conf_mat_class_score_save_dir = target_dir + '/confusion_mat_class_score/'

    
    files_of_interest = [ source_dir + "/" + 
                    "alex_multi_7_withConf_result_data.json"]
    result_file_name = "alex_multi_7_color_noMedFilt"
   
   

    #uncertainty_thresh_vals = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.12]
    #uncertainty_thresh_vals = np.linspace(0.0001, 0.12, 100)
    uncertainty_thresh_vals = np.linspace(0.00002, 0.12, 100)
    uncertainty_thresh_vals_sparse = np.linspace(0.00002, 0.12, 10)
    total_class_score_thresh_vals = np.linspace(0.3, 0.9, 7)
    
    all_results = load_result_files(files_of_interest)
    merged_result = merge_results(all_results)
    results_np = convert_results_to_np(merged_result)
    correct_ind, wrong_ind = check_predictions(results_np)
    pred_class_scores = get_pred_class_score(results_np)
    pred_class_var = get_pred_class_var(results_np)
    
    # Calculates the mean of the variance of predicted class for cases with
    # correct and wrong predictions
    correct_pred_var_mean = np.mean(pred_class_var[correct_ind]) 
    wrong_pred_var_mean = np.mean(pred_class_var[wrong_ind])
    
    print("correct_pred_var_mean: ", correct_pred_var_mean)
    print("wrong_pred_var_mean: ", wrong_pred_var_mean)
    
    # Draw plots of accuracy vs. uncertainty threshold
    #draw_acc_vs_uncertainty(results_np, uncertainty_thresh_vals,
                            #graph_save_dir, result_file_name)
    
    # Plot and save confusion matrices for different uncertainty threshold
    # values
    #for thresh in uncertainty_thresh_vals_sparse:
        #draw_confusion_mat_confident_res(results_np, thresh,
                        #result_file_name + '_' + '{0:.3f}'.format(thresh),
                        #conf_mat_save_dir)
        
    for thresh in total_class_score_thresh_vals:
        mask = filter_on_class_score(results_np, thresh)
        title = "Class score thresh: {0:.3f}".format(thresh)
        print("Mask size: ", np.sum(mask))
        draw_confusion_mat_with_mask(results_np, mask,
                          result_file_name + '_' + '{0:.3f}'.format(thresh),
                          conf_mat_class_score_save_dir, 
                          title)
   
    
    #save_results(target_dir, "merged_multi_results", merged_result)
    
    
