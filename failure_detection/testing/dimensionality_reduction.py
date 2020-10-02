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
import numpy.matlib
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
from tqdm import tqdm


if __name__ == "__main__":
    # base_dir = ("/data/CAML/IVOA_CRA/evaluation_multi_class_uncertainty")
    #target_dir = source_dir + '/embeddings/'
    # target_dir = base_dir + '/clustering/'

    target_dir = ("/data/CAML/IVOA_ML_toolv2/dataset_000/evaluation_multi_class_uncertainty/alex_locked_a_last_model_017/clustering_r_4c_unc.005_NoSub_pca100/")

    clustering_res_path = (target_dir + 
        '/clustered_embeddings_kmeans.pkl')
   
    result_file_name = (
      "clustered_embeddings")

    # Runs dimensionality reduction before clustering
    APPLY_DIM_REDUCTION = True
    n_pca_components = 50

    SUBSAMPLE_PER_CLUSTER = True
    SUBSAMPLING_RATIO = 0.05 # def: 0.05
    MIN_CLUSTER_SAMPLE = 500
    
    #********************
    #*** Loading data
    
    file_cluster = open(clustering_res_path, 'rb')
    clustering, selected_embeddings = pickle.load(file_cluster)
    file_cluster.close()

    # *********************
    # **** Subsampling ****
    # *********************

    # Remove the index file for subsampled data if it exists from
    # previous runs of this scripts
    patch_indices_path = (target_dir+result_file_name
                            +'_patch_indices_vis_subsample.csv')
    if os.path.exists(patch_indices_path):
        os.remove(patch_indices_path)

    if SUBSAMPLE_PER_CLUSTER:
        # cluster_num = clustering.cluster_centers_.shape[0]
        # for i in range(cluster_num):
        #     cluster_indices[i] = np.argwhere(clustering.labels_ == i)

        cluster_num = clustering.cluster_centers_.shape[0]
        cluster_indices = {}
        cluster_samples = np.zeros((0, 0), dtype=int)
        # cluster_samples = np.zeros((VIS_SAMPLE_NUM, cluster_num), dtype=int)
        for i in range(cluster_num):
            cluster_indices[i] = np.argwhere(clustering.labels_ == i)
            
            # Give high weight to points close to the cluster center
            cluster_size = cluster_indices[i].size
            cluster_ctr = clustering.cluster_centers_[i,:]
            datapoints = selected_embeddings
            dist_to_ctr = (datapoints[cluster_indices[i].flatten(), :] - 
                        np.matlib.repmat(cluster_ctr, cluster_size, 1))
            dist_to_ctr_norm = np.linalg.norm(dist_to_ctr, axis=1)
            sampling_weight = 1.0 / (0.1 + dist_to_ctr_norm)
            sampling_weight = sampling_weight / np.sum(sampling_weight)
            
            sample_num = floor(SUBSAMPLING_RATIO * cluster_indices[i].size)
            sample_num = max(MIN_CLUSTER_SAMPLE , sample_num)
            sample_num = min(cluster_indices[i].size, sample_num)
            new_cluster_samples = np.random.choice(cluster_indices[i].flatten(),
                                                   replace=False,
                                                   size=sample_num, 
                                                   p=sampling_weight)
            cluster_samples = np.append(cluster_samples, new_cluster_samples)
            print('Cluster# ', i, ' subsampled: ', sample_num, '/', 
                  cluster_indices[i].size)

        selected_embeddings = selected_embeddings[cluster_samples, :]
        print('selected_embeddings: ', selected_embeddings.shape)

        patch_indices_path = (target_dir+result_file_name
                             +'_patch_indices_vis_subsample.csv')
        np.savetxt(patch_indices_path, cluster_samples, delimiter=",",
                   fmt='%i')
        print('Subsampled datapoints indices were saved to file.')

    # *****************
    # **** Run PCA ****
    # *****************
    if APPLY_DIM_REDUCTION:
        pca_50 = PCA(n_components=n_pca_components)
        pca_result_50 = pca_50.fit_transform(selected_embeddings)
        print ('Cumulative explained variation for {} principal '
            'components:{}'.format(n_pca_components, 
                            np.sum(pca_50.explained_variance_ratio_)))
        
        print('PCA result size: ', pca_result_50.shape)
    else:
        pca_result_50 = selected_embeddings
    
    
    # *****************
    # *** Run t-SNE ***
    # *****************

    # def compute_distance_matrix():
    #     dist = np.zeros((pca_result_50.shape[0], pca_result_50.shape[0]))
    #     for i in tqdm(range(pca_result_50.shape[0]-1)):
    #         for j in range(i+1, pca_result_50.shape[0]):
    #             dist[i][j] =  np.linalg.norm(pca_result_50[i] - pca_result_50[j])
    #             if clustering.labels_[i] != clustering.labels_[j]:
    #                 dist[i][j] *= 4
    #             dist[j][i] = dist[i][j]
    #     return dist

    clustering_label_lookup = {}
    for i in range(pca_result_50.shape[0]):
        clustering_label_lookup[pca_result_50[i].tobytes()] = clustering.labels_[i]

    def compute_distance(x, y):
        dist = np.linalg.norm(x - y)
        if x.tobytes() not in clustering_label_lookup or y.tobytes() not in clustering_label_lookup:
            return dist
        if (clustering_label_lookup[x.tobytes()] != clustering_label_lookup[y.tobytes()]):
            dist *= 4
        return dist
        
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, metric=compute_distance, n_jobs=16)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, n_jobs=16)

    tsne_results = tsne.fit_transform(pca_result_50)
    
    tSNE_save_path = target_dir + result_file_name +'_tsne_res.csv'
    np.savetxt(tSNE_save_path, 
               tsne_results)
    print('Saved the tSNE result in ', tSNE_save_path)
