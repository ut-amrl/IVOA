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
from tqdm import tqdm


if __name__ == "__main__":
    base_dir = ("/data/CAML/IVOA_CRA/evaluation_multi_class_uncertainty")
    #target_dir = source_dir + '/embeddings/'
    target_dir = base_dir + '/clustering/'

    clustering_res_path = (target_dir + 
        'embeddings/clustered_embeddings_kmeans.pkl')
   
    result_file_name = (
      "clustered_embeddings")
    
    #********************
    #*** Loading data
    
    file_cluster = open(clustering_res_path, 'rb')
    clustering, selected_embeddings = pickle.load(file_cluster)
    file_cluster.close()
    # *****************
    # **** Run PCA ****
    # *****************
    n_components = 50
    pca_50 = PCA(n_components=n_components)
    pca_result_50 = pca_50.fit_transform(selected_embeddings)
    print ('Cumulative explained variation for {} principal '
          'components:{}'.format(n_components, 
                          np.sum(pca_50.explained_variance_ratio_)))
    
    print('PCA result size: ', pca_result_50.shape)
    
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
        
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, metric=compute_distance, n_jobs=16)
    tsne_results = tsne.fit_transform(pca_result_50)
    
    tSNE_save_path = target_dir + result_file_name +'_tsne_res.csv'
    np.savetxt(tSNE_save_path, 
               tsne_results)
    print('Saved the tSNE result in ', tSNE_save_path)
