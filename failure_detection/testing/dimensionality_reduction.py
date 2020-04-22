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
    # Base directory used for loading saved results and also setting the
    # target path
    base_dir = ("/media/ssd2/results/IVOA/initial_results/")

    # Target directory for saving the results
    target_dir = base_dir + '/embeddings/'
    # target_dir = source_dir + '/embeddings2048/'

    clustering_res_path = (base_dir + '/embeddings/' + 
        'alex_multi_7_newIndoor_noMedFilt_PCA20_thresh03_bwp01_subp3_meanshift.pkl')

    result_file_name = 'alex_multi_7_newIndoor_noMedFilt_PCA20_thresh03_bwp01_subp3_meanshift'

    #********************
    #### Parameters
    #********************
    
    
    #********************
    #*** Loading data
    
    file_cluster = open(clustering_res_path, 'rb')
    clustering, selected_embeddings = pickle.load(file_cluster)

    # *****************
    # **** Run PCA ****
    # *****************
    n_components = 20
    pca_50 = PCA(n_components=n_components)
    pca_result_50 = pca_50.fit_transform(selected_embeddings)
    print ('Cumulative explained variation for {} principal '
          'components:{}'.format(n_components, 
                          np.sum(pca_50.explained_variance_ratio_)))
    
    print('PCA result size: ', pca_result_50.shape)
    
    # *****************
    # *** Run t-SNE ***
    # *****************
    def cluster_based_distance(x, y):
        import pdb; pdb.set_trace()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, metric=cluster_based_distance)
    tsne_results = tsne.fit_transform(pca_result_50)
    
    
    tSNE_save_path = target_dir + result_file_name +'_tsne_res.csv'
    np.savetxt(tSNE_save_path, 
               tsne_results)
    print('Saved the tSNE result in ', tSNE_save_path)
    

    
    
