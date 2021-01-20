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

# This script loads the data image by image as opposed to patch by patch.
# This allows it to process each image by passing overlapping patches
# through the network and then predict the class for each query points
# using the information from a window around the query points
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
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
from test_multi_class_model import calculate_gpu_usage
from test_multi_class_model import plot_confusion_matrix
from test_multi_class_model import save_results
from evaluator import Evaluator
from drawing import annotate_results_on_image


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
    parser.add_argument('--model_type', default=None, 
                        help="alex_bin, alex_multi, obs_detection",
                        required=True)
    parser.add_argument('--data_path', default=None, 
                        help="Path to test data",
                        required=True)
    parser.add_argument('--meta_data_path', default=None, 
                        help="Path to the meta data (ground truth labels)",
                        required=True)
    parser.add_argument('--model_dir', default=None, 
                        help='The path to load the trained model from.', 
                        required=True)
    parser.add_argument('--save_dir', default=None, 
                        help='The path to save the images', 
                        required=True)
    parser.add_argument('--result_file_name', default=None, 
                        help='Name of the result file (will be saved in pdf.)', 
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
    parser.add_argument('--calc_uncertainty', default=None,
                  type=lambda s: s.lower() in ['true','t','yes', '1'],
                  help='True if you want to calculate uncertainty values',
                  required=True)
    parser.add_argument('--patch_crop_size', default=None, 
                        help=('The size the of original patch before being '
                              'resized to fit the network input size.'), 
                        type=int, required=True)
    parser.add_argument('--workers_num', default=None, 
                        help=('Number of threds to be used for data loading.'
                              ' It should be set as twice the number of CPUs'), 
                        type=int, required=False)
    parser.add_argument('--stride_size', default=None, 
                        help=('stride size when processing the full input image'
                              ), 
                        type=int, required=True)
    parser.add_argument('--eval_bbox_size', default=None, 
                        help=('Size of the bounding box around a query point '
                              'that is processed for estimating a filtered ' 
                              'prediction for that point'), 
                        type=int, required=True)
    parser.add_argument('--segment_id', default=None, 
                        help=('The segment id to be processed in the bagfile'
                          '. Only applicable if a single bagfile is '
                          'requested to be processed.'), 
                        type=int, required=False)
    parser.add_argument('--segment_len', default=None, 
                        help=('Length of the segment to be processed in the '
                          'bagfile. Only applicable if a single bagfile is '
                          'requested to be processed.'), 
                        type=int, required=False)
    parser.add_argument('--bagfile_id', default=None, 
                        help=('Bagfile to be processed. This argument only'
                          ' accepts a single bagfile number.'), 
                        type=int, required=False)
    
    args = parser.parse_args()
    
    MODEL_TYPE = args.model_type
    MODEL_LOAD_DIR = args.model_dir
    RESULTS_SAVE_DIR = args.save_dir
    RESULT_FILE_NAME = args.result_file_name
    LOAD_MULTI_GPU_MODEL = args.use_multi_gpu_model
    USE_GPU = args.use_gpu
    root_dir = args.data_path
    META_DATA_DIR = args.meta_data_path
    USE_COLOR_IMAGES = args.use_color_images
    CALC_UNCERTAINTY = args.calc_uncertainty
    PATCH_SIZE = args.patch_crop_size
    EVAL_PIXEL_STRIDE = args.stride_size
    EVAL_BBOX_SIZE = args.eval_bbox_size
    
    #SEGMENT_ID = args.segment_id if args.segment_id else None
    #SEGMENT_LEN = args.segment_len if args.segment_len else None
    
    SEGMENT_ID = args.segment_id
    SEGMENT_LEN = args.segment_len
    
    #EVAL_PIXEL_STRIDE = 100 # stride size when processing the full input image
    #EVAL_BBOX_SIZE = 100 # size of the bounding box around a query point
                        # that is processed for estimating a filtered 
                        # prediction for that point
    SAVE_ALL_BAG_RESULTS_TOGETHER = True # If this is set to false, only the
                        # per bagfile prediction and GT results will be saved
                        # to file
    ONLINE_VISUALIZATION = False
    SAVE_VISUALIZATIONS = True
    SAVE_PROCESSED_IMAGES = True # saves the class probabilities and uncertainty
                                # (heatmaps) as separate files to be used for 
                                # post processecing
    PARTICLE_NUM = 20 # 50, 20
    EVAL_BATCH_SIZE = 1 #100 90
    FIGURE_FOLDER_HEATMAP= 'heatmaps'
    FIGURE_FOLDER_PROC_IMG = 'processed_imgs'
    PER_BAG_RESULTS_FOLDER = 'bagfile_results' 
    PER_SEGMENT_RESULTS_FOLDER = 'segment_results'
    AVAILABLE_MODELS = ['alex_bin', 'alex_multi', 'obs_detection']
    USE_COLORING_MASK = True
    USE_MULTI_GPU = True
    DATASET_IMAGE_CHANNEL = 3 # number of channels of the dataset
    BATCH_SIZE = 1
    NUM_WORKERS = 2 
   
    

    bagfile_list_test = [0]
    
    # Overwrite the above values if they are set as inputs
    if args.workers_num is not None:
        NUM_WORKERS = args.workers_num
    if args.bagfile_id:
        bagfile_list_test = [args.bagfile_id]
 
    if not(MODEL_TYPE in AVAILABLE_MODELS):
        print(MODEL_TYPE, " not available. Choose from ", AVAILABLE_MODELS)
        exit()
    
    # ***********************************
    # **** Execution and model information
    print("Loaded model: ", MODEL_LOAD_DIR)
    print("Save dir: ", RESULTS_SAVE_DIR)
    print("Result file name: ", RESULT_FILE_NAME)
    print("Test data: ", bagfile_list_test)
    print("Full image batch size: ", BATCH_SIZE)
    print("Image patch batch size (EVAL_BATCH_SIZE): ", EVAL_BATCH_SIZE)
    print("Eval stride: ", EVAL_PIXEL_STRIDE)
    print("Eval bbox size: ", EVAL_BBOX_SIZE)
    print("Workers num: ", NUM_WORKERS)
    
    if CALC_UNCERTAINTY:
        print("Uncertainty values are calculated with a particle size of ",
              PARTICLE_NUM)


    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        used_gpu_count = 1
        total_mem =( 
             float(torch.cuda.get_device_properties(device).total_memory) 
             / 1000000.0)
        gpu_name = torch.cuda.get_device_name(device)
        print("Using ", gpu_name, " with ", total_mem, " MB of memory.")
    else:
        device = torch.device("cpu")
        used_gpu_count = 0
        
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
                                            full_img_transform,
                                            meta_data_dir = META_DATA_DIR,
                                            load_patch_info = True,
                                            segment_id = SEGMENT_ID,
                                            segment_length = SEGMENT_LEN)

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
        classes = ["perception_true_pos", "perception_true_neg", 
                  "perception_false_pos", "perception_false_neg"]
        
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
        
    if USE_GPU and USE_MULTI_GPU:
      if torch.cuda.device_count() > 1:
          used_gpu_count = torch.cuda.device_count()
          print("Using ", used_gpu_count, " GPUs.")
          net = nn.DataParallel(net)
        
    net = net.to(device)
   
    evaluator = Evaluator(net, EVAL_PIXEL_STRIDE, PATCH_SIZE, (600, 960),
                          patch_transform, device, classes, EVAL_BATCH_SIZE,
                    particle_size= PARTICLE_NUM if CALC_UNCERTAINTY else 1,
                    eval_bbox_size=EVAL_BBOX_SIZE)
    
    # Calculate the checkpoints in terms of the number of mini batches such
    # that there exist 10 checkpoints per epoch
    dataset_size = len(test_dataset)
    checkpoint_num = floor(float(dataset_size / BATCH_SIZE) / 10)
    checkpoint_num = max([checkpoint_num, 1])
    
    print("Starting Inference...")
    start_time = time.time()
  
    for cur_phase in phases:
        net.eval()   # Set model to evaluate mode
        
        running_corrects = 0
        class_count = np.zeros(4)
        all_predictions = np.array([])
        all_class_probs = np.array([])
        all_class_var = np.array([])
        all_binary_labels = np.array([])
        all_multi_class_labels = np.array([])
       
        prev_bagfile_name = ""
        bagfile_start_idx = 0
        # Iterate over data
        for i, data in enumerate(data_loaders[cur_phase], 0):
            #class_count += count_class_instances(data)
            #print('Class count: ', class_count)
            
            # get the inputs
            full_img = data['full_img']
            img_name = data['full_img_name']
            bagfile_name = data['bagfile_name']
            #inputs = inputs.to(device)
            patch_coords_left = data['patch_coords_left']
            jpp_obs_existence = data['jpp_obs_existence']
            kinect_obs_existence = data['kinect_obs_existence']
            multi_class_labels = data['multi_class_label']
          
           
            evaluator.process_image(full_img[0, :, :, :])
            predictions = evaluator.predict(patch_coords_left[0,:,:].numpy())
            
            all_predictions = np.append(all_predictions, predictions)
            all_multi_class_labels = np.append(all_multi_class_labels,
                                               multi_class_labels)
            
            #all_class_probs = np.append(all_class_probs, , 0)
            #all_class_var = np.append(all_class_var, , 0)
            
            if SAVE_VISUALIZATIONS:
                save_path = (RESULTS_SAVE_DIR + '/pred_visualization/' +
                             bagfile_name[0])
                annotate_results_on_image(evaluator.raw_image, 
                                          patch_coords_left[0,:,:].numpy(), 
                                          np.asarray(multi_class_labels), 
                                          predictions,
                                          classes, 
                                          save_path, 
                                          img_name[0])
            if SAVE_VISUALIZATIONS or ONLINE_VISUALIZATION:
                evaluator.generate_heatmap(visualize=ONLINE_VISUALIZATION, 
                                           save=SAVE_VISUALIZATIONS,
                save_dir=(RESULTS_SAVE_DIR + '/' + FIGURE_FOLDER_HEATMAP 
                          + '/' + bagfile_name[0]),
                          file_name=img_name[0])
            if SAVE_PROCESSED_IMAGES:
                save_dir = (RESULTS_SAVE_DIR + '/' + FIGURE_FOLDER_PROC_IMG 
                          + '/' + bagfile_name[0])
                evaluator.save_processed_images(save_dir,
                                                img_name[0][:-4])
            


            if i % checkpoint_num == checkpoint_num - 1:  # print every 
                                            # checkpoint_num 1000 mini-batches
                print('%d%%: Processed %d datapoints.' 
                      % (((i + 1) * BATCH_SIZE * 100.00) / dataset_size, 
                         (i + 1)*BATCH_SIZE))
                if used_gpu_count:
                    print("Total GPU usage (MB): ",
                          calculate_gpu_usage(used_gpu_count), " / ",
                          used_gpu_count * total_mem)
            
            # Save the predictions results of each bagfile separately to
            # keep a snapshot of the progress
            if (prev_bagfile_name != bagfile_name[0]) and (i > 0):
                cnf_matrix = confusion_matrix(
                          all_multi_class_labels[bagfile_start_idx:],
                          all_predictions[bagfile_start_idx:])
                plot_confusion_matrix(
                    cnf_matrix, 
                    classes,
                    RESULT_FILE_NAME + bagfile_name[0], 
                    RESULTS_SAVE_DIR + '/' + PER_BAG_RESULTS_FOLDER,
                    normalize=True)
                
                save_results(RESULTS_SAVE_DIR+ '/' + PER_BAG_RESULTS_FOLDER, 
                             RESULT_FILE_NAME+ bagfile_name[0], 
                      classes, 
                      all_multi_class_labels[bagfile_start_idx:], 
                      all_predictions[bagfile_start_idx:],
                      np.array([]).reshape(0,len(classes)), 
                      np.array([]).reshape(0,len(classes)))
                
                bagfile_start_idx = i + 1
            
            prev_bagfile_name = bagfile_name[0]

                
        print('All data was processed.')
        time_elapsed = time.time() - start_time
        print('Completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed//3600, (time_elapsed % 3600)//60, time_elapsed % 60))
        
        print('Acc: %f'% accuracy_score(all_multi_class_labels, 
                              all_predictions,
                              normalize = True))
        
        if SAVE_ALL_BAG_RESULTS_TOGETHER:
            # If running only a segment of a bagfile, do not draw the 
            # confusion matrix
            if (SEGMENT_ID is None) or (SEGMENT_LEN is None):
                cnf_matrix = confusion_matrix(all_multi_class_labels,
                                      all_predictions)
                plot_confusion_matrix(
                    cnf_matrix, 
                    classes,
                    RESULT_FILE_NAME, RESULTS_SAVE_DIR,
                    normalize=True)
            
            if (SEGMENT_ID is not None) and (SEGMENT_LEN is not None):
                result_dir = (RESULTS_SAVE_DIR + '/' + 
                      PER_SEGMENT_RESULTS_FOLDER + '/' + bagfile_name[0] + '/')
                result_file ='{:05d}_{:08d}_{:s}'.format(
                              SEGMENT_ID,SEGMENT_LEN,RESULT_FILE_NAME)
            else:
                result_dir = RESULTS_SAVE_DIR
                result_file = RESULT_FILE_NAME
                
            save_results(result_dir, result_file, 
                  classes, 
                  all_multi_class_labels, all_predictions,
                  np.array([]).reshape(0,len(classes)), 
                  np.array([]).reshape(0,len(classes)))

