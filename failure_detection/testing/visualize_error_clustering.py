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
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import numpy.matlib
import pandas as pd
import time
import json
import copy
import argparse
from skimage import io, transform
from torchvision import transforms, utils
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib import cm
from math import floor
from collections import OrderedDict
from data_loader.load_patches import FailureDetectionDataset
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_pdf import PdfPages
from analyze_results import *


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
          event.x, event.y, event.xdata, event.ydata))
          
def show_img(img, cluster_idx, save=True, member_count=None):
    npimg = img.numpy()
    fig = plt.figure()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    if save:
        if member_count is not None:
            suffix = '_cluster_{}_memberCount_{}.png'.format(cluster_idx, 
                        member_count)
        else:
            suffix = '_cluster_{0}.png'.format(cluster_idx)
        plt.imsave(target_dir + '/' + result_name  + suffix, 
                   np.transpose(npimg, (1,2,0)))
        # plt.imsave(target_dir + '/' + result_name  + '_cluster_{0}.png'.format(cluster_idx), np.transpose(npimg, (1,2,0)))

def generate_colors(color_num):
  start = 0.0
  stop = 1.0
  cm_subsection = np.linspace(start, stop, color_num) 

  colors = [ cm.tab20b(x) for x in cm_subsection ]
  return colors        
    
if __name__ == "__main__":
  
    #   LABEL_COLOR_MAP = {0 : 'b',
    #                      1 : 'g',
    #                      2 : 'r',
    #                      3 : 'cyan',
    #                      4 : 'magenta',
    #                      5 : 'yellow',
    #                      6 : 'black'}
  
      source_dir = ("/data/CAML/IVOA_ML_toolv2/dataset_000/evaluation_multi_class_uncertainty/alex_locked_a_last_model_017")
      target_dir = source_dir + '/clustering_r_4c_unc.005_NoSub_pca100'
      result_name = 'kmeans'

      #********************
      # Modify the following paths
      # ********************

      #****************************
      #********* Subsampled Mean MeanShift
      tsne_res_path = (target_dir + '/clustered_embeddings_tsne_res.csv')
      tsne_patch_indices_path = (target_dir + '/clustered_embeddings_patch_indices.csv')
      clustering_res_path = (target_dir + '/clustered_embeddings_kmeans.pkl')

      dataset_path = "/data/CAML/IVOA_ML_toolv2/dataset_000/"

      #----------------------------------
      # Parameters
      
      CLUSTER_IN_ORIGINAL_SPACE = True # If set to true the result of
                                    # clustering in the original space 
                                    # will be loaded
      CLUSTERING_METHOD = 'MeanShift' # {'kmeans','dbscan', 'MeanShift'}
      
      CLUSTER_NUM = 3
      
      VIS_SAMPLE_NUM = 50 # 50
      VIS_GRID_ROW = 5
      VIS_GRID_COL = 10 # 10
      
      PATCH_SIZE = 50
      
      sessions_list = [13, 17, 20, 22, 23]
      
      device = "cpu"

      # Colors assigned to cluster numbers
    #   LABEL_COLOR_MAP = {0: 'b',
    #                      1: 'g',
    #                      2: 'r',
    #                      3: 'cyan',
    #                      4: 'magenta',
    #                      5: 'yellow',
    #                      6: 'black'}

      #-----------------------------------
      # Initializations

      # This will be set to True if existing index files indicate that 
      # during dimensionality reduction (dimensionality_reduction.py) datapoints
      # have been subsampled.
      dim_reduction_with_subsamp = False

      data_transform = transforms.Compose([
          transforms.CenterCrop(PATCH_SIZE),
          transforms.ToTensor()
      ])
     
      
      # Initialize the dataset for image extraction and visualization
      test_dataset = FailureDetectionDataset(dataset_path, 
                            sessions_list,
                            3,
                            3,
                            data_transform,
                            meta_data_dir = dataset_path,
                            extract_from_full_img = True,
                            patch_size = PATCH_SIZE)
      
      
      # Load the saved results of t-SNE
      tsne_res = np.loadtxt(tsne_res_path, dtype=float)

      # Check if datapoints have been further subsampled during execution of
      # dimensionality_reduction.py and load the corresponding patch index file.
      tsne_subsampled_indices_path = os.path.join(target_dir, 
                    'clustered_embeddings_patch_indices_vis_subsample.csv')
      if os.path.exists(tsne_subsampled_indices_path):
        print('Patch index file {} was found. \nLoading indices of datapoints '
               'subsampled by dimensionality_reduction.py'.format(tsne_subsampled_indices_path))
        tsne_sub_samp_patch_ind = np.loadtxt(tsne_subsampled_indices_path, dtype=int)
        dim_reduction_with_subsamp = True
      
      tsne_patch_ind = np.loadtxt(tsne_patch_indices_path, dtype=int)
      if dim_reduction_with_subsamp:
        tsne_patch_ind = tsne_patch_ind[tsne_sub_samp_patch_ind]


      if CLUSTER_IN_ORIGINAL_SPACE:
          if CLUSTERING_METHOD == 'kmeans':
              #reload object from file
              file_cluster = open(clustering_res_path, 'rb')
              clustering, selected_embeddings = pickle.load(file_cluster)
              file_cluster.close()
              CLUSTER_NUM = clustering.cluster_centers_.shape[0]
              cluster_centers = clustering.cluster_centers_
          elif CLUSTERING_METHOD == 'dbscan':
              #reload object from file
              file_cluster = open(clustering_res_path, 'rb')
              clustering, selected_embeddings = pickle.load(file_cluster)
              file_cluster.close()
              CLUSTER_NUM = clustering.core_sample_indices_.shape[0]
              cluster_centers = selected_embeddings[
                                          clustering.core_sample_indices_, :]
              print("cluster centers:")
              print(cluster_centers.shape)
              noisy_labels = clustering.labels_ == -1
              print("Noisy label num: ", np.sum(noisy_labels))
              n_clusters_ = (len(set(clustering.labels_)) - 
                              (1 if -1 in clustering.labels_ else 0))
              print("Cluster num: ", n_clusters_)
              print(clustering.labels_)
              print(clustering.core_sample_indices_)
          elif CLUSTERING_METHOD == 'MeanShift':
              #reload object from file
              file_cluster = open(clustering_res_path, 'rb')
              clustering, selected_embeddings = pickle.load(file_cluster)
              file_cluster.close()
              CLUSTER_NUM = clustering.cluster_centers_.shape[0]
              cluster_centers = clustering.cluster_centers_
              print("cluster centers:")
              print(cluster_centers.shape)
          else:
              print('Unknown clustering method ', CLUSTERING_METHOD)
              exit()
      else:
          # Apply clustering on the output of t-SNE
          #clustering = KMeans(n_clusters=CLUSTER_NUM, 
                              #random_state=0).fit(tsne_res)
          clustering = DBSCAN(eps=100, min_samples=50, 
                            n_jobs=4, algorithm='brute').fit(tsne_res)
      
      if dim_reduction_with_subsamp:
          selected_embeddings = selected_embeddings[tsne_sub_samp_patch_ind]
          clustering.labels_ = clustering.labels_[tsne_sub_samp_patch_ind]
     
      # Setup event callback for the figure
      fig, ax = plt.subplots()
      #cid = fig.canvas.mpl_connect('button_press_event', onclick)
     

      
      LABEL_COLOR_MAP = generate_colors(CLUSTER_NUM)
      
      # Plot the result of clustering on the t-SNE output
      fig, ax = plt.subplots() 
      for i in range(CLUSTER_NUM):
          indices = clustering.labels_ == i
          label_color = LABEL_COLOR_MAP[i]
          label = str(i)
          plt.scatter(tsne_res[indices,0], tsne_res[indices,1], 
                      color=label_color, 
                      label=label)
      plt.title("K-Means clustering result")
      plt.legend()
      
      #--------------------------------------
      # Randomly sample points from each cluster and visualize the
      # corresponding image patches
      cluster_indices = {}
      cluster_samples = np.zeros((VIS_SAMPLE_NUM, CLUSTER_NUM), dtype=int)
      for i in range(CLUSTER_NUM):
        cluster_indices[i] = np.argwhere(clustering.labels_ == i)
        
        # Give high weight to points close to the cluster center
        cluster_size = cluster_indices[i].size
        cluster_ctr = cluster_centers[i,:]
        if CLUSTER_IN_ORIGINAL_SPACE:
            datapoints = selected_embeddings
        else:
            datapoints = tsne_res
        dist_to_ctr = (datapoints[cluster_indices[i].flatten(), :] - 
                    np.matlib.repmat(cluster_ctr, cluster_size, 1))
        dist_to_ctr_norm = np.linalg.norm(dist_to_ctr, axis=1)
        # print('************************')
        # print('max dist: ', np.max(dist_to_ctr_norm))
        # print('min dist: ', np.min(dist_to_ctr_norm))
        min_dist = np.min(dist_to_ctr_norm)
        max_dist = np.max(dist_to_ctr_norm)
        dist_range = max_dist - min_dist
        dist_to_ctr_norm = (dist_to_ctr_norm - min_dist)
        if dist_range > 0.00001:
            dist_to_ctr_norm /= dist_range

        sampling_weight = 1.0 / (0.01 + dist_to_ctr_norm * dist_to_ctr_norm)
        
        sampling_weight = sampling_weight / np.sum(sampling_weight)
        sample_with_replacement = False
        if VIS_SAMPLE_NUM > cluster_size:
            print('NOTICE: There exist only {} datapoints in cluster# {}'.format
                    (cluster_size, i))
            sample_with_replacement = True

        cluster_samples[:,i] = np.random.choice(cluster_indices[i].flatten(),
                            size=VIS_SAMPLE_NUM, p=sampling_weight, replace=sample_with_replacement)

      
      # Plot the sampled points on each cluster
      fig2, ax = plt.subplots() 
      for i in range(CLUSTER_NUM):
          indices = clustering.labels_ == i
          label_color = LABEL_COLOR_MAP[i]
          label = str(i)
          plt.scatter(tsne_res[indices,0], tsne_res[indices,1], 
                      color=label_color, 
                      label=label)
          plt.scatter(tsne_res[cluster_samples[:,i],0],
                      tsne_res[cluster_samples[:,i],1], c='black')
      plt.title("K-Means clustering result")
      plt.legend()
      
      
      # Visualize the image patches
      for i in range(CLUSTER_NUM):
          patch_indices = tsne_patch_ind[cluster_samples[:,i]]
          
          patch_list = torch.tensor([], dtype=torch.float,
                                    device = device)
          for j in patch_indices:
              data = test_dataset[j]
              patch = data['patch']
              patch = patch.reshape(1, patch.shape[0], patch.shape[1], 
                                    patch.shape[2])
              patch_list = torch.cat((patch_list, patch),0)
              
          member_count = None
          cluster_size = cluster_indices[i].size
          if VIS_SAMPLE_NUM > cluster_size:    
              member_count = cluster_size
          show_img(make_grid(patch_list, nrow=VIS_GRID_ROW), i, 
                   member_count=member_count)
          #full_img = transforms.ToPILImage()(full_img[0, :, :, :])
          #full_img = full_img.convert(mode = "RGB")
      

      fig.savefig(target_dir + '/' + result_name + '_fig1.png')
      fig2.savefig(target_dir + '/' + result_name  + '_fig2.png')

      
      
      
