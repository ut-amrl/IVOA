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
import time
import json
import copy
import argparse
from skimage import io, transform
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from math import floor
from collections import OrderedDict
from data_loader.load_patches import FailureDetectionDataset
from networks.failure_detection_multi_class import FailureDetectionMultiClassNet
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages


# Helper function for saving inference results
def save_results(target_dir, file_name, classes, 
                 ground_truth, predictions, class_prob, class_uncertainty):
    
    # Handle the case when there is no class_uncertainty available
    formatted_class_uncertainty = {}
    if(class_uncertainty.size != 0):
        formatted_class_uncertainty = {
                classes[x]: class_uncertainty[:,x].tolist() 
                              for x in range(len(classes))}
    formatted_class_prob = {}
    if(class_prob.size != 0):
        formatted_class_prob = {
                classes[x]: class_prob[:,x].tolist() 
                              for x in range(len(classes))}

    gt_int = [int(x) for x in ground_truth.tolist()]
    results = {"classes": classes,
              "ground_truth": gt_int,
              "predictions": predictions.tolist(),
              "class_prob": formatted_class_prob,
              "class_uncertainty": formatted_class_uncertainty}
    results_json = json.dumps(results, indent=2)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file = open(target_dir +'/'+file_name+'_data.json', 'w');
    file.write(results_json)
    file.close()

        
# Helper function to draw the confusion matrix
def plot_confusion_matrix(cm, classes, file_name, file_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure("confusion_mat")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    pp = PdfPages(file_path + '/' + file_name + '.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.close("confusion_mat")
    
def calculate_gpu_usage(gpu_num):
    total_usage_mb = 0.0
    for i in range(gpu_num):
        total_usage_mb += float(torch.cuda.max_memory_allocated(i) +
                          torch.cuda.max_memory_cached(i)) / 1000000.0
    return total_usage_mb

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process input arguments.')
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
                        help='The path to save the results.', 
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
    parser.add_argument('--online_patch_extraction', default=None,
                  type=lambda s: s.lower() in ['true','t','yes', '1'],
                  help='True if you want to extract patches from full images',
                  required=True)
    parser.add_argument('--patch_crop_size', default=None, 
                        help=('The size the of original patch before being '
                              'resized to fit the network input size.'), 
                        type=int, required=True)
    parser.add_argument('--test_set', default=None, 
                        help="Test set name",
                        required=True)
    
    args = parser.parse_args()
   
    MODEL_LOAD_DIR = args.model_dir
    RESULTS_SAVE_DIR = args.save_dir
    RESULT_FILE_NAME = args.result_file_name
    LOAD_MULTI_GPU_MODEL = args.use_multi_gpu_model
    USE_GPU = args.use_gpu
    root_dir = args.data_path
    META_DATA_DIR = args.meta_data_path
    USE_COLOR_IMAGES = args.use_color_images
    CALC_UNCERTAINTY = args.calc_uncertainty
    ONLINE_PATCH_EXTRACTION = args.online_patch_extraction
    PATCH_SIZE = args.patch_crop_size
    TEST_SET_NAME = args.test_set
    
    PARTICLE_NUM = 50 # number of passes for each input image def:50
    DATASET_IMAGE_CHANNEL = 3 # number of channels of the dataset
    LOAD_MODEL_WEIGHTS = True
    USE_MULTI_GPU = True
    #BATCH_SIZE = 2000 # 400
    #NUM_WORKERS = 12 # Allocate 4 GPUs and 6 Cpus
    BATCH_SIZE = 2 # 400
    NUM_WORKERS = 4 # Allocate 4 GPUs and 6 Cpus
    
    test_set_dict = {
      "test_1":[0]
    }
   
    
    if (TEST_SET_NAME is not None):
        bagfile_list_test = test_set_dict[TEST_SET_NAME]
        RESULT_FILE_NAME = RESULT_FILE_NAME+"_"+TEST_SET_NAME
    
    print("Loaded model: ", MODEL_LOAD_DIR)
    print("Result file name: ", RESULT_FILE_NAME)
    print("Test data: ", bagfile_list_test)
    print("Batch size: ", BATCH_SIZE)
    print("Workers num: ", NUM_WORKERS)
    
    if CALC_UNCERTAINTY:
        print("Uncertainty values are calculated with a particle size of ",
              PARTICLE_NUM)
    
    class_names = ["jpp_true_pos", "jpp_true_neg", "jpp_false_pos", 
                  "jpp_false_neg"]
    
    if not os.path.exists(RESULTS_SAVE_DIR):
          os.makedirs(RESULTS_SAVE_DIR)
    
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
    
    # Scales pixel intensities to [0, 1]
    scale_intensities = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                             std=[255.0, 255.0, 255.0])
    # Normalizes the images to [-1 ,1]. The mean and std values are the
    # standard values used for pretrained torchvision models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    
    # If the input to the network is mono images, concatenate the single 
    # channel 3 times to make a placeholder RGB for alexnet
    if USE_COLOR_IMAGES:
        data_transform = transforms.Compose([
            transforms.CenterCrop(PATCH_SIZE),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        data_transform = transforms.Compose([
            transforms.CenterCrop(PATCH_SIZE),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            normalize
        ])
   
    output_image_channel = 3 if USE_COLOR_IMAGES else 1
    test_dataset = FailureDetectionDataset(root_dir, 
                            bagfile_list_test,
                            DATASET_IMAGE_CHANNEL,
                            output_image_channel,
                            data_transform,
                            meta_data_dir = META_DATA_DIR,
                            extract_from_full_img = ONLINE_PATCH_EXTRACTION,
                            patch_size = PATCH_SIZE)

    datasets = {phases[0]: test_dataset}
    
    data_loaders = {x: torch.utils.data.DataLoader(datasets[x], 
                                                   batch_size=BATCH_SIZE,
                                                   num_workers=NUM_WORKERS)
                    for x in phases}
    
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
    
    
    if USE_GPU and USE_MULTI_GPU:
      if torch.cuda.device_count() > 1:
          used_gpu_count = torch.cuda.device_count()
          print("Using ", used_gpu_count, " GPUs.")
          net = nn.DataParallel(net)
          
    net = net.to(device)
    
    # Calculate the checkpoints in terms of the number of mini batches such
    # that there exist 10 checkpoints per epoch
    dataset_size = len(test_dataset)
    checkpoint_num = floor(float(dataset_size / BATCH_SIZE) / 10)
    checkpoint_num = max([checkpoint_num, 1])
    
    print("Starting Inference...")
    start_time = time.time()
    
    for cur_phase in phases:
        net.eval()   # Set model to evaluate mode
        
        # Turn on dropout during evaluation for uncertainty estimation
        if CALC_UNCERTAINTY:
            for mod in net.modules():
                if mod.__class__.__name__.startswith('Dropout'):
                    mod.train()
        
        running_corrects = 0
        class_count = np.zeros(4)
        all_predictions = torch.tensor([], dtype=torch.long, device=device)
        all_class_probs = torch.tensor([], dtype=torch.float, device=device)
        all_class_var = torch.tensor([], dtype=torch.float, device=device)
        all_binary_labels = np.array([])
        all_multi_class_labels = np.array([])
        
        softmax_layer = nn.Softmax(dim=1)
        
        # Iterate over data
        for i, data in enumerate(data_loaders[cur_phase], 0):
            #class_count += count_class_instances(data)
            #print('Class count: ', class_count)
           
            
            # get the inputs
            inputs = data['patch']
            binary_labels = data['binary_label']
            multi_class_labels = data['multi_class_label']
            inputs = inputs.to(device)
            labels = multi_class_labels.to(device)
            
            particle_class_probs = torch.tensor([], dtype=torch.float,
                                                device = device)
            # Pass the same patch multiple times through the network if 
            # uncertainty calculation is required
            if CALC_UNCERTAINTY:
                for j in range(PARTICLE_NUM):
                
                    # track history if only in train
                    with torch.set_grad_enabled(False):
                        outputs = net(inputs)
                        outputs_softmax = softmax_layer(outputs)
                        outputs_softmax = torch.reshape(outputs_softmax, 
                           (outputs_softmax.shape[0], len(class_names), 1))
                        particle_class_probs = torch.cat((particle_class_probs,
                                                          outputs_softmax), 2)  
 
   
                
                # Calculates the mean class scores and the prediction variance
                # over all particles
                mean_class_prob = torch.mean(particle_class_probs, 2)
                _, mean_preds = torch.max(mean_class_prob, 1)
                predicted_class_var = torch.var(particle_class_probs, 2)
                
                all_class_probs = torch.cat((all_class_probs, mean_class_prob)
                                            , 0)
                all_class_var = torch.cat((all_class_var, predicted_class_var)
                                            , 0)
            
            # Normal operation with no uncertainty calculation (it is
            # separated from the CALC_UNCERTAINTY mode for runtime efficiency)
            else:
                # track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = net(inputs)
                    _, mean_preds = torch.max(outputs, 1)
                    
                all_class_probs = torch.cat((all_class_probs, 
                                            softmax_layer(outputs))
                                            , 0)

            
            all_predictions = torch.cat((all_predictions, mean_preds), 0)
            all_binary_labels = np.append(all_binary_labels, 
                                          binary_labels)
            all_multi_class_labels = np.append(all_multi_class_labels, 
                                              multi_class_labels)
            

            corrects = torch.sum(mean_preds == labels)
            running_corrects += corrects

            
            
            if i % checkpoint_num == checkpoint_num - 1:  # print every 
                                            # checkpoint_num 1000 mini-batches
                print('%d%%: Processed %d datapoints.' 
                      % (((i + 1) * BATCH_SIZE * 100.00) / dataset_size, 
                         (i + 1)*BATCH_SIZE))
                if used_gpu_count:
                    print("Total GPU usage (MB): ",
                          calculate_gpu_usage(used_gpu_count), " / ",
                          used_gpu_count * total_mem)
        
        all_predictions_np = all_predictions.to(torch.device("cpu")).numpy()
        all_class_probs_np = all_class_probs.to(torch.device("cpu")).numpy()
        all_class_var_np = all_class_var.to(torch.device("cpu")).numpy()
        print('All data was processed.')
        time_elapsed = time.time() - start_time
        print('Completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed//3600, (time_elapsed % 3600)//60, time_elapsed % 60))
        print('Acc: %f'% accuracy_score(all_multi_class_labels, 
                              all_predictions_np,
                              normalize = True))
        cnf_matrix = confusion_matrix(all_multi_class_labels,
                              all_predictions_np)
        plot_confusion_matrix(
            cnf_matrix, 
            ["jpp_true_pos", "jpp_true_neg", "jpp_false_pos", "jpp_false_neg"],
            RESULT_FILE_NAME, RESULTS_SAVE_DIR,
            normalize=True)

        save_results(RESULTS_SAVE_DIR, RESULT_FILE_NAME, 
                     class_names, 
                      all_multi_class_labels, all_predictions_np,
                      all_class_probs_np, all_class_var_np)
        


