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
import json
import copy
from PIL import Image, ImageDraw
import argparse
from skimage import io, transform
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from math import floor, ceil
from collections import OrderedDict
from data_loader.load_full_images import FailureDetectionDataset
from networks.failure_detection_multi_class import FailureDetectionMultiClassNet
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages

# Add visualization to the image for binary introspective model
def color_and_save_image(img, pred_labels, patch_size, save_path, img_name,
                         use_mask, coloring_mask):
    img_copy = img.copy()
    img_draw = ImageDraw.Draw(img)
    for j in range(pred_labels.shape[0]):
        for k in range(pred_labels.shape[1]):
            if use_mask and (coloring_mask[j][k] == 0):
                continue
            
            color = 'red'
            if patch_pred_labels[j, k]:
                # The case when the network has predicted for JPP to be correct
                color = 'green'
            else:
                # The case when the network has predicted for JPP to be wrong
                color = 'red'
                    
            img_draw.rectangle((k * patch_size, j * patch_size, 
                        (k+1) * patch_size, (j+1) * patch_size), 
                                outline=color, 
                                fill=color)

            
    out_img = Image.blend(img_copy, img, 0.3)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory " , save_path ,  " Created ")

    out_img.save(save_path + '/' + img_name, format='JPEG')

# Add visualization to the image for obstacle detection model (binary model 
# trained on Kinect ground truth)  
def color_and_save_image_obs_det(img, pred_labels, patch_size, 
                                 save_path, img_name):
    img_copy = img.copy()
    img_draw = ImageDraw.Draw(img)
    for j in range(pred_labels.shape[0]):
        for k in range(pred_labels.shape[1]):
            
            color = 'orange'
            if patch_pred_labels[j, k]:
                # The case when the network has detected an obstacle
                color = 'orange'
            else:
                # The case when the network has detected traversable terrain
                color = 'blue'
                    
            img_draw.rectangle((k * patch_size, j * patch_size, 
                        (k+1) * patch_size, (j+1) * patch_size), 
                                outline=color, 
                                fill=color)
            
    out_img = Image.blend(img_copy, img, 0.3)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory " , save_path ,  " Created ")

    out_img.save(save_path + '/' + img_name, format='JPEG')

# Add visualization to the image for multi_class introspective model that
# predicts instances of true_positive, true_negative, false_positive, 
# and false_negative for JPP
# Background green color means that IVOA predicts a correct prediction by 
# the percpetion model under test (TP or TN)
# Red circle in a patch means that IVOA predicts that the model under test
# will predict an obstacle for that patch
def color_and_save_image_multi_class(img, pred_labels, 
                                     patch_size, save_path, 
                                     img_name):
    img_copy = img.copy()
    img_draw = ImageDraw.Draw(img)
    for j in range(pred_labels.shape[0]):
        for k in range(pred_labels.shape[1]):
            
            color = 'red'
            obstacle_detected = False
            if patch_pred_labels[j, k] == 0:
                # True positive
                color = 'green'
                obstacle_detected = True
            elif patch_pred_labels[j, k] == 1:
                # True negative
                color = 'green'
            elif patch_pred_labels[j, k] == 2:
                # False positive
                color = 'red'
                obstacle_detected = True
            else:
                # False negative
                color = 'red'
                    
            img_draw.rectangle((k * patch_size, j * patch_size, 
                        (k+1) * patch_size, (j+1) * patch_size), 
                                outline=color, 
                                fill=color)
            
            if obstacle_detected:
                img_draw.ellipse((k * patch_size, j * patch_size, 
                             (k+1) * patch_size, (j+1) * patch_size), 
                                outline='black', 
                                fill=color)
            
    out_img = Image.blend(img_copy, img, 0.3)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory " , save_path ,  " Created ")

    out_img.save(save_path + '/' + img_name, format='JPEG')
    
def create_color_mask(width, height, step_size, width_ratio, height_ratio):
    width_bin = floor(width / step_size)
    height_bin = floor(height / step_size)
    mask = np.zeros((height_bin, width_bin))
    w_ind_st = floor(width_bin / 2.0 - (width_ratio * width_bin)/2)
    h_ind_st = floor(height_bin - height_ratio * height_bin)
    w_ind_end = ceil(width_bin / 2.0 + (width_ratio * width_bin)/2)
    h_ind_end = floor(height_bin)
    
    print([w_ind_st, w_ind_end, h_ind_st, h_ind_end])
    
    for i in range(h_ind_st, h_ind_end):
        for j in range(w_ind_st, w_ind_end):
            mask[i][j] = 1
    
    print(mask)
    return mask
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--data_path', default=None, 
                        help="Path to test data",
                        required=True)
    parser.add_argument('--model_dir', default=None, 
                        help='The path to load the trained model from.', 
                        required=True)
    parser.add_argument('--save_dir', default=None, 
                        help='The path to save the images', 
                        required=True)
    parser.add_argument('--use_multi_gpu_model', default=None,
                        type=lambda s: s.lower() in ['true','t','yes', '1'],
                        help='Is the model trained using multiple GPUs',
                        required=True)
    parser.add_argument('--use_gpu', default=None,
                        type=lambda s: s.lower() in ['true','t','yes', '1'],
                        help='If you want to use GPUs.',
                        required=True)
    parser.add_argument('--use_color_images', default=None,
                  type=lambda s: s.lower() in ['true','t','yes', '1'],
                  help='True if you want to feed the network with color images',
                  required=True)
    parser.add_argument('--patch_crop_size', default=None, 
                        help=('The size the of original patch before being '
                              'resized to fit the network input size.'), 
                        type=int, required=True)
    parser.add_argument('--test_set', default=None, 
                        help="Test set name",
                        required=True)
    
    args = parser.parse_args()
    
    MODEL_TYPE = "alex_multi"
    MODEL_LOAD_DIR = args.model_dir
    RESULTS_SAVE_DIR = args.save_dir
    LOAD_MULTI_GPU_MODEL = args.use_multi_gpu_model
    USE_GPU = args.use_gpu
    root_dir = args.data_path
    USE_COLOR_IMAGES = args.use_color_images
    PATCH_SIZE = args.patch_crop_size
    TEST_SET_NAME = args.test_set
  
    AVAILABLE_MODELS = ['alex_bin', 'alex_multi', 'obs_detection']
    USE_COLORING_MASK = True
    DATASET_IMAGE_CHANNEL = 3 # number of channels of the dataset
    BATCH_SIZE = 1
    NUM_WORKERS = 6 # Allocate 4 GPUs and 6 Cpus
    
    test_set_dict = {
      "test1":[0]
    }
    
    if (TEST_SET_NAME is not None):
        bagfile_list_test = test_set_dict[TEST_SET_NAME]
 
    if not(MODEL_TYPE in AVAILABLE_MODELS):
        print(MODEL_TYPE, " not available. Choose from ", AVAILABLE_MODELS)
        exit()


    if USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("device: ", device)
    phases = ['test']
    data_set_portion_to_sample = {'test': 1}
    
    # Use height ratios 0.5 and 0.6 for a small and large mask respectively
    if USE_COLORING_MASK:
        coloring_mask = create_color_mask(960, 600, PATCH_SIZE, 
                                      width_ratio = 0.6,
                                      height_ratio = 0.6)
    
    # Scales pixel intensities to [0, 1]
    scale_intensities = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                             std=[255.0, 255.0, 255.0])
    # Normalizes the images to [-1 ,1]. The mean and std values are the
    # standard values used for pretrained torchvision models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    

    patch_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    full_img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
   
    
    output_image_channel = 3 if USE_COLOR_IMAGES else 1
    test_dataset = FailureDetectionDataset(root_dir, 
                                            bagfile_list_test,
                                            DATASET_IMAGE_CHANNEL,
                                            output_image_channel,
                                            full_img_transform)

    datasets = {phases[0]: test_dataset}
    
    data_loaders = {x: torch.utils.data.DataLoader(datasets[x], 
                                                   batch_size=BATCH_SIZE,
                                                   num_workers= NUM_WORKERS)
                    for x in phases}
    
    if MODEL_TYPE in ['alex_bin', 'obs_detection']:
        print("Not supported!")
        exit()
    elif MODEL_TYPE == 'alex_multi':
        net = FailureDetectionMultiClassNet()
        
    ## If the model has been saved using nn.DataParallel, the names
    ## of the keys of the model dictionary start with an extra 'module.'
    ## which should be removed
     
    # Map to CPU as you load the model
    state_dict = torch.load(MODEL_LOAD_DIR, 
                  map_location=lambda storage, loc: storage)
    was_trained_with_multi_gpu = False
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            was_trained_with_multi_gpu = True
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
            print(name)
    
    if(was_trained_with_multi_gpu):
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state_dict)
        
    net = net.to(device)
  
    for cur_phase in phases:
        net.eval()   # Set model to evaluate mode
        
        running_corrects = 0
        class_count = np.zeros(4)
        all_predictions = np.array([])
        all_binary_labels = np.array([])
        all_multi_class_labels = np.array([])
        # Iterate over data
        for i, data in enumerate(data_loaders[cur_phase], 0):
            #class_count += count_class_instances(data)
            #print('Class count: ', class_count)
            
            # get the inputs
            full_img = data['full_img']
            img_name = data['full_img_name']
            bagfile_name = data['bagfile_name']
            #inputs = inputs.to(device)
            
            full_img = transforms.ToPILImage()(full_img[0, :, :, :])
            full_img = full_img.convert(mode = "RGB")


          
            #img_height = full_img.shape[2]
            #img_width = full_img.shape[3]
            img_height = full_img.size[1]
            img_width = full_img.size[0]
            patch_pred_labels = np.zeros((floor(img_height / PATCH_SIZE),
                                         floor(img_width / PATCH_SIZE)))
            for j in range(0, floor(img_height / PATCH_SIZE)):
                for k in range(0, floor(img_width / PATCH_SIZE)):
                    patch = transforms.functional.crop(full_img,
                                                    j * PATCH_SIZE, 
                                                    k * PATCH_SIZE,
                                                    PATCH_SIZE, PATCH_SIZE)
                    #patch.show()
                    #print(patch)
                    #input("Press [enter] to continue.")
                    patch_transformed = patch_transform(patch)
                    patch_transformed = patch_transformed.to(device)
                    patch_transformed = patch_transformed.reshape(
                                        1,
                                        patch_transformed.shape[0],
                                        patch_transformed.shape[1],
                                        patch_transformed.shape[2])
                    
                    # track history if only in train
                    with torch.set_grad_enabled(False):
                        # forward + backward + optimize
                        outputs = net(patch_transformed)
                        _, preds = torch.max(outputs, 1)
                        patch_pred_labels[j, k] = preds.cpu().numpy()
            
            if MODEL_TYPE == 'alex_bin':
                color_and_save_image(full_img, patch_pred_labels, PATCH_SIZE, 
                                 RESULTS_SAVE_DIR + '/' + bagfile_name[0],
                                 img_name[0],
                                 USE_COLORING_MASK, coloring_mask)
            elif MODEL_TYPE == 'obs_detection':
                color_and_save_image_obs_det(full_img, patch_pred_labels, 
                                 PATCH_SIZE, 
                                 RESULTS_SAVE_DIR + '/' + bagfile_name[0],
                                 img_name[0])
            elif MODEL_TYPE == 'alex_multi':
                color_and_save_image_multi_class(full_img, patch_pred_labels, 
                                 PATCH_SIZE, 
                                 RESULTS_SAVE_DIR + '/' + bagfile_name[0],
                                 img_name[0])



            if i % 10000 == 9999:    # print every 1000 mini-batches
                print('Processed %d datapoints.' % (i + 1)*BATCH_SIZE)

                
        print('All data was processed.')
