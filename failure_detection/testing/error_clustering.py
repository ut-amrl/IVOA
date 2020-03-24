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
          
def show_img(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


if __name__ == "__main__":
  
      LABEL_COLOR_MAP = {0 : 'b',
                         1 : 'g',
                         2 : 'r',
                         3 : 'cyan',
                         4 : 'magenta',
                         5 : 'yellow',
                         6 : 'black'}
  
      source_dir = ("/hdd/results/introspective_failure_detection/"
                  "alex_multi_7_color_noMedFilt")
    
      target_dir = source_dir + '/embeddings/'
      
      files_of_interes_patch_info = [ source_dir + "/" + 
        "alex_multi_7__withConf_result_newIndoor_patch_info.json"]


      #****************************
      #********* Subsampled Mean MeanShift
      tsne_res_path = (source_dir + '/embeddings/' + 
      
'alex_multi_7_newIndoor_noMedFilt_PCA20_thresh03_bwp01_subp3_tsne_res.csv')
      tsne_patch_indices_path = (source_dir + '/embeddings/' + 
 
'alex_multi_7_newIndoor_noMedFilt_PCA20_thresh03_bwp01_subp3_patch_indices.csv'
)
      clustering_res_path = (source_dir + '/embeddings/' + 
     
'alex_multi_7_newIndoor_noMedFilt_PCA20_thresh03_bwp01_subp3_meanshift.pkl')
      
      
      
      
      dataset_path = "/hdd/datasets"\
                    "/introspective_failure_detection_noMedFilt_colored" \
      
      #----------------------------------
      # Parameters
      
      CLUSTER_IN_ORIGINAL_SPACE = True # If set to true the result of
                                    # clustering in the original space 
                                    # will be loaded
      CLUSTERING_METHOD = 'MeanShift' # {'kmeans','dbscan', 'MeanShift'}
      
      CLUSTER_NUM = 3
      
      VIS_SAMPLE_NUM = 50
      VIS_GRID_ROW = 5
      VIS_GRID_COL = 10
      
      PATCH_SIZE = 100
      
      bagfile_list = [1]
      
      device = "cpu"
      
      #-----------------------------------
      # Initializations
      
      data_transform = transforms.Compose([
          transforms.CenterCrop(PATCH_SIZE),
          transforms.ToTensor()
      ])
     
      
      # Initialize the dataset for image extraction and visualization
      test_dataset = FailureDetectionDataset(dataset_path, 
                            bagfile_list,
                            3,
                            3,
                            data_transform,
                            meta_data_dir = dataset_path,
                            extract_from_full_img = True,
                            patch_size = PATCH_SIZE)
      
      
      # Load the saved results of t-SNE
      tsne_res = np.loadtxt(tsne_res_path, dtype=float)
      tsne_patch_ind = np.loadtxt(tsne_patch_indices_path, dtype=int)

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
      
     
     
      # Setup event callback for the figure
      fig, ax = plt.subplots()
      #cid = fig.canvas.mpl_connect('button_press_event', onclick)
     

      # Plot the output of t-SNE
      plt.scatter(tsne_res[:,0], tsne_res[:,1])
      plt.title("t-SNE Output")
      
      
      # Plot the result of clustering on the t-SNE output
      label_color = [LABEL_COLOR_MAP[l] for l in clustering.labels_]
      plt.scatter(tsne_res[:,0], tsne_res[:,1], c=label_color)
      plt.title("K-Means clustering result")
      
      
      
      
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
          sampling_weight = 1.0 / (0.1 + dist_to_ctr_norm)
          sampling_weight = sampling_weight / np.sum(sampling_weight)
          
          cluster_samples[:,i] = np.random.choice(cluster_indices[i].flatten(),
                              size=VIS_SAMPLE_NUM, p=sampling_weight)
      
      # Plot the sampled points on each cluster
      fig2, ax = plt.subplots()
      label_color = [LABEL_COLOR_MAP[l] for l in clustering.labels_]
      plt.scatter(tsne_res[:,0], tsne_res[:,1], c=label_color)
      plt.title("K-Means clustering result")
      
      for i in range(CLUSTER_NUM):
          plt.scatter(tsne_res[cluster_samples[:,i],0],
                      tsne_res[cluster_samples[:,i],1], c='black')
      
      
      
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
              
          show_img(make_grid(patch_list, nrow=VIS_GRID_ROW))
          #full_img = transforms.ToPILImage()(full_img[0, :, :, :])
          #full_img = full_img.convert(mode = "RGB")
      
      
      plt.show()
      
      
      
