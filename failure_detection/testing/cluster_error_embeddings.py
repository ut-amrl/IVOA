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

import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import time
import json
import copy
import argparse, os
from skimage import io, transform
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from math import floor
from collections import OrderedDict
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from analyze_results import *


if __name__ == "__main__":
    source_dir = ("/data/CAML/IVOA_CRA/evaluation_multi_class_uncertainty")

    target_dir = source_dir + '/clustering'
    
    # **** 2048 embedding size:
    files_of_interes_patch_info = [ source_dir + "/embeddings/embeddings" + 
        "embeddings_test_1_patch_info.json"]
    files_of_interest_embeddings = [ source_dir + "/embeddings/embeddings" +
        "embeddings_test_1_patch_embeddings.csv"]
    files_of_interest_prediction = [ source_dir + "/" + 
        "evaluation_multi_class_uncertainty_test_1_data.json"]
    
    result_file_name = (
      "/clustered_embeddings")
        
    #********************
    #### Parameters
    UNCERTAINTY_THRESH = 0.03
    CLUSTER_NUM = 3
    CLUSTERING_METHOD = 'kmeans' # {'kmeans','dbscan', 'MeanShift'}
    
    SUBSAMPLE_DATA = True
    SUBSAMPLING_RATIO = 0.05
    
    #********************
    #*** Loading data
    prediction_results = load_result_files(files_of_interest_prediction)
    merged_pred_results = merge_results(prediction_results)
    pred_results_np = convert_results_to_np(merged_pred_results)
   
    embeddings = np.array([], dtype=float)
    for i in range(len(files_of_interest_embeddings)):
        file = files_of_interest_embeddings[i]
        curr_embed = np.loadtxt(file, dtype=float, delimiter=',')
        if i == 0:
            embeddings = curr_embed
        else:
            embeddings = np.append(embeddings, curr_embed, 0)
    
    print("Size of embeddings: ", embeddings.shape)
    print(type(embeddings))
    
    # Keep only data points classified as FP and FN
    FN_mask = pred_results_np["predictions"] == 3 
    FP_mask = pred_results_np["predictions"] == 2 
    perception_error_mask = np.logical_or(FN_mask, FP_mask)
    print("Number of FN and FP instances: " ,np.sum(perception_error_mask))
    
    # Keep only data points that are predicted with confidence
    confident_mask = filter_unconfident(pred_results_np,
                                        UNCERTAINTY_THRESH) # 0.03, 0.02, 
                                                                # 0.15, 0.015
    final_mask = np.logical_and(perception_error_mask, confident_mask)
    print("Number of confident FN and FP instances: " 
           ,np.sum(final_mask))
    final_mask_indices = np.argwhere(final_mask)
        
    if SUBSAMPLE_DATA:
        data_num = final_mask_indices.size
        sample_num = floor(data_num * SUBSAMPLING_RATIO)
        sampled_indices = np.random.choice(final_mask_indices.flatten(), 
                                    replace=False, size=sample_num)
        subsampled_mask = np.zeros_like(final_mask)
        subsampled_mask[sampled_indices] = 1
        final_mask_indices = sampled_indices
        final_mask = subsampled_mask
        print("Subsampled data num: ", final_mask_indices.size)
        print("final mask size: ", np.sum(final_mask))
    
    # Save the indices of the selected patches
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    patch_indices_path = (target_dir+result_file_name+'_patch_indices.csv')
    np.savetxt(patch_indices_path, final_mask_indices, delimiter=",",
               fmt='%i')
    print('Patch indices were saved to file.')
    
    selected_embeddings = embeddings[final_mask, :]
    print("Size of selected embeddings: ", selected_embeddings.shape)
    
    # *****************
    # **** Clustering**
    # *****************
    if CLUSTERING_METHOD == 'kmeans':
        # Apply clustering in the embedding space
        kmeans =(KMeans(n_clusters=CLUSTER_NUM,random_state=0).fit(
                selected_embeddings))
        
        # Save the kmeans object to file
        file_path = target_dir + result_file_name + '_kmeans.pkl'
        afile = open(file_path, 'wb')
        pickle.dump([kmeans, selected_embeddings], afile)
        afile.close()
    elif CLUSTERING_METHOD == 'dbscan':
        clustering = DBSCAN(eps=50, min_samples=10, 
                            n_jobs=4).fit(selected_embeddings)
        # Save the dbscan object to file
        file_path = target_dir + result_file_name + '_dbscan.pkl'
        afile = open(file_path, 'wb')
        pickle.dump([clustering, selected_embeddings], afile)
        afile.close()
    elif CLUSTERING_METHOD == 'MeanShift':
        clustering = MeanShift(bandwidth=0.07).fit(selected_embeddings)
        # Save the dbscan object to file
        file_path = target_dir + result_file_name + '_meanshift.pkl'
        afile = open(file_path, 'wb')
        pickle.dump([clustering, selected_embeddings], afile)
        afile.close()
    print('Saved the result of the clustering in embedding space.')
    print('Clustering method: ', CLUSTERING_METHOD)