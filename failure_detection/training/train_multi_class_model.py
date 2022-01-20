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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def count_class_instances(data):
    jpp_obs = np.asarray(data['jpp_obs_existence'])
    kinect_obs = np.asarray(data['kinect_obs_existence'])
    # True positive
    true_pos = ((jpp_obs == kinect_obs) & kinect_obs)
    
    # True negative
    true_neg = ((jpp_obs == kinect_obs) & np.logical_not(kinect_obs))
    
    # False positive
    false_pos = ((jpp_obs != kinect_obs) & np.logical_not(kinect_obs))
    
    # False negative
    false_neg = ((jpp_obs != kinect_obs) & kinect_obs)
    
    class_count = np.array([true_pos.sum(), true_neg.sum(),
                   false_pos.sum(), 
                   false_neg.sum()])
    return class_count
    
# Helper function for saving training results
def save_results(target_dir, model_name, last_model, best_model, train_history,
                 snapshot_idx):
    suffix = ''
    if (snapshot_idx >= 0):
        suffix = '_' + str('%03d' % snapshot_idx)
    torch.save(last_model.state_dict(), 
               target_dir + model_name + '_last_model' + suffix + '.pt')
    torch.save(best_model, 
               target_dir + model_name + '_best_model' + suffix + '.pt')
    hist_file_path = (target_dir + model_name + '_training_history' + 
                     suffix + '.json')
    with open(hist_file_path, 'w') as fp:
        json.dump(train_history, fp, indent=2)
        
def calculate_gpu_usage(gpu_num):
    total_usage_mb = 0.0
    for i in range(gpu_num):
        total_usage_mb += float(torch.cuda.max_memory_allocated(device) +
                          torch.cuda.max_memory_cached(device)) / 1000000.0
    return total_usage_mb

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--data_path', default=None, 
                        help="Path to training data",
                        required=True)
    parser.add_argument('--meta_data_path', default=None, 
                        help="Path to the meta data (ground truth labels)",
                        required=True)
    parser.add_argument('--model_save_dir', default=None, 
                        help='The path to save the trained model in.', 
                        required=True)
    parser.add_argument('--snapshot_save_dir', default=None, 
                        help='The path to save the snapshots of trained model', 
                        required=True)
    parser.add_argument('--model_name', default=None, 
                        help='Name of model to be saved.', 
                        required=True)
    parser.add_argument('--use_color_images', default=None,
                  type=lambda s: s.lower() in ['true','t','yes', '1'],
                  help='True if you want to feed the network with color images',
                  required=True)
    parser.add_argument('--online_patch_extraction', default=None,
                  type=lambda s: s.lower() in ['true','t','yes', '1'],
                  help='True if you want to extract patches from full images',
                  required=True)
    parser.add_argument('--patch_crop_size', default=None, 
                        help=('The size the of original patch before being '
                              'resized to fit the network input size.'), 
                        type=int, required=True)
    parser.add_argument('--fn_sample_weight_coeff', default=1.0, 
                        help=('Multiplies by the sampling weight for the false'
                              ' negative class. Use for outdoor data.'), 
                        type=float, required=False)
    parser.add_argument('--fp_sample_weight_coeff', default=1.0, 
                        help=('Multiplies by the sampling weight for the false'
                              ' positive class.'), 
                        type=float, required=False)
    parser.add_argument('--tp_sample_weight_coeff', default=1.0, 
                        help=('Multiplies by the sampling weight for the true'
                              ' positive class.'), 
                        type=float, required=False)
    parser.add_argument('--tn_sample_weight_coeff', default=1.0, 
                        help=('Multiplies by the sampling weight for the true'
                              ' negative class.'), 
                        type=float, required=False)
    parser.add_argument('--image_scale_factor', default=1.0, 
                        help=('Scale factor for resizing the loaded images.'),
                        type=float, required=False)
    parser.add_argument('--train_set', default=None, 
                        help="Training set name",
                        required=True)
    parser.add_argument('--validation_set', default=None, 
                        help="Validation set name",
                        required=True)
    
    args = parser.parse_args()
    
    
    MODEL_SAVE_DIR = args.model_save_dir
    SNAPSHOT_SAVE_DIR = args.snapshot_save_dir
    MODEL_NAME = args.model_name
    root_dir = args.data_path
    META_DATA_DIR = args.meta_data_path
    USE_COLOR_IMAGES = args.use_color_images
    ONLINE_PATCH_EXTRACTION = args.online_patch_extraction
    PATCH_SIZE = args.patch_crop_size
    TRAIN_SET_NAME = args.train_set
    VALIDATION_SET_NAME = args.validation_set
    
    class_weights_coeff = [args.tp_sample_weight_coeff, args.tn_sample_weight_coeff,
                           args.fp_sample_weight_coeff, args.fn_sample_weight_coeff]

    NETWORK_MODEL = "alexnet" # "alexnet", "resnet152", "inception_v3"
    LOCK_FEATURE_EXT_LAYERS = False
    USE_GPU = True
    USE_MULTI_GPU = True
    LOAD_MULTI_GPU_MODEL = True
    DATASET_IMAGE_CHANNEL = 3 # number of channels of the dataset images
    LOAD_MODEL_WEIGHTS = False

    MODEL_LOAD_DIR =(
                  "/data/CAML/IVOA_CRA/models/snapshot/"
                  "cra_full_train_model_unlocked_best_model_010.pt")
   
  
    EPOCH_NUM = 30
    SNAPSHOT_FREQ = 2 # take a snapshot once every 2 epochs
    #BATCH_SIZE = 400 # 
    #NUM_WORKERS = 12 # Allocate 4 GPUs and 6 Cpus
    #NUM_WORKERS = 6 # Allocate 4 GPUs and 6 Cpus
    
    #BATCH_SIZE = 1000 #  default until train_6
    #NUM_WORKERS = 24 # Allocate 8 GPUs and 12 Cpus
    #NUM_WORKERS = 2 # Allocate 8 GPUs and 12 Cpus

    # BATCH_SIZE = 4096 #
    # NUM_WORKERS = 16 #

    BATCH_SIZE = 1000 #
    NUM_WORKERS = 10 #
    
    # BATCH_SIZE = 400 #
    # NUM_WORKERS = 8 #

    valid_set_dict = {
      "valid_1":[1003],
      "valid_1_cpip_tr0_v1":[1007, 1008],
      "valid_2_cpip_tr1_v0_husky":[9, 11, 7, 23, 27, 35],
      "valid_1_ganet_tr0_v0":[1007]
    }

    train_set_dict = {
      "train_1":[1001, 1004],
      "train_1_cpip_tr0_v1":[1000, 1001, 1002, 1003, 1004, 1005, 1006],
      "train_2_cpip_tr1_v0_husky":list(set(range(0,45)).difference(valid_set_dict["valid_2_cpip_tr1_v0_husky"])),
      "train_1_ganet_tr0_v0":[1008]
    }
   
    
    if (TRAIN_SET_NAME is not None) and (VALIDATION_SET_NAME is not None):
        bagfile_list_train = train_set_dict[TRAIN_SET_NAME]
        bagfile_list_val = valid_set_dict[VALIDATION_SET_NAME]
        
    
    if LOAD_MODEL_WEIGHTS: print("Loaded Model: ", MODEL_LOAD_DIR)
    print("Output Model: ", MODEL_NAME)
    print("Train data: ", bagfile_list_train)
    print("Validation data: ", bagfile_list_val)
    print("Network base model is ", NETWORK_MODEL)
    print("Batch size: ", BATCH_SIZE)
    print("Workers num: ", NUM_WORKERS)
    
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
    phases = ['train', 'val']
    #data_set_portion_to_sample = {'train': 0.8, 'val': 0.2}
    data_set_portion_to_sample = {'train': 0.9, 'val': 0.2}
    
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
    train_dataset = FailureDetectionDataset(root_dir, 
                            bagfile_list_train,
                            DATASET_IMAGE_CHANNEL,
                            output_image_channel,
                            data_transform,
                            meta_data_dir = META_DATA_DIR,
                            extract_from_full_img = ONLINE_PATCH_EXTRACTION,
                            patch_size = PATCH_SIZE,
                            image_scale_factor = args.image_scale_factor,
                            class_weights_coeff = class_weights_coeff)
    val_dataset = FailureDetectionDataset(root_dir,
                            bagfile_list_val,
                            DATASET_IMAGE_CHANNEL,
                            output_image_channel,
                            data_transform,
                            meta_data_dir = META_DATA_DIR,
                            extract_from_full_img = ONLINE_PATCH_EXTRACTION,
                            patch_size = PATCH_SIZE,
                            image_scale_factor = args.image_scale_factor,
                            class_weights_coeff = class_weights_coeff)
    datasets = {phases[0]: train_dataset, phases[1]: val_dataset}
    
    # Uses a weighted random sampler to oversample the minority classes and
    # undersample the majority classes    
    data_info = {x: datasets[x].get_dataset_info() for x in phases}
    sample_weights = {x: data_info[x]['sample_weights'].tolist() 
                      for x in phases}
    
    # Number of samples to draw from train and validation dataset in each epoch
    sample_nums = {x:floor(data_set_portion_to_sample[x] * 
                     len(sample_weights[x])) 
                   for x in phases}
    print('sample nums: ', sample_nums)
    
    samplers = {x: WeightedRandomSampler(sample_weights[x],
                                        sample_nums[x],
                                        replacement = True)
                for x in phases}
    
    data_loaders = {x: torch.utils.data.DataLoader(datasets[x], 
                                                   batch_size=BATCH_SIZE,
                                               num_workers=NUM_WORKERS, 
                                               sampler=samplers[x])
                    for x in phases}
   
    net = FailureDetectionMultiClassNet(base_model=NETWORK_MODEL,
                                      lock_feature_ext=LOCK_FEATURE_EXT_LAYERS)

    if LOAD_MODEL_WEIGHTS:
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
 
    
    if USE_MULTI_GPU:
      if torch.cuda.device_count() > 1:
          used_gpu_count = torch.cuda.device_count()
          print("Using ", used_gpu_count, " GPUs.")
          net = nn.DataParallel(net)

    net = net.to(device)
    
    # Gathering the parameters to be updated
    params_to_update = []
    print("Params to learn:")
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=0.005, momentum=0.9)
    
    #optimizer = optim.Adam(params_to_update, lr=0.001, betas=(0.9, 0.999), 
                                    #eps=1e-08, weight_decay=0, amsgrad=False)
    #print("WARNING: ADAM optimizer is being used********")
    
    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())
    training_history = {x: {'loss': [], 'acc': []} 
                        for x in phases}

    print("Starting Training...")
    start_time = time.time()   
 
    # Runs training and validation
    for epoch in range(EPOCH_NUM):
        for cur_phase in phases:
            if cur_phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_corrects = 0
            running_loss = 0.0
            running_corrects = 0
            class_count = np.zeros(4)
            # Iterate over data
            for i, data in enumerate(data_loaders[cur_phase], 0):
                #class_count += count_class_instances(data)
                #print('Class count: ', class_count)
                
                # get the inputs
                inputs = data['patch']
                labels = data['multi_class_label']
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(cur_phase == 'train'):
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # backward and optimize only if in training phase
                    if cur_phase == 'train':
                        loss.backward()
                        optimizer.step()


                # print statistics
                loss = loss.item()
                corrects = torch.sum(preds == labels)
                running_loss += loss
                epoch_loss += loss
                running_corrects += corrects
                epoch_corrects += corrects
               

                if cur_phase == 'train':
                    if i % 1000 == 999:    # print every 1000 mini-batches
                        print('[%d, %5d] Loss: %.6f  Acc: %.6f' %
                            (epoch + 1, i + 1, running_loss / 1000,
                            running_corrects.double() / (1000.0 * BATCH_SIZE)))
                        running_corrects = 0
                        running_loss = 0.0
                        
                        if used_gpu_count:
                            print("Total GPU usage (MB): ",
                              calculate_gpu_usage(used_gpu_count), " / ",
                              used_gpu_count * total_mem)
                        
            epoch_loss = epoch_loss / (sample_nums[cur_phase] / BATCH_SIZE)
            epoch_acc = epoch_corrects.double() / sample_nums[cur_phase]
            print('%s: Loss: %.6f  Acc: %.6f' % 
                  (cur_phase, epoch_loss, epoch_acc))
            
            # Keep the best model so far
            if cur_phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(net.state_dict())
            
            training_history[cur_phase]['loss'].append(epoch_loss)
            epoch_acc_float = epoch_acc.cpu().numpy().tolist()
            training_history[cur_phase]['acc'].append(epoch_acc_float)
        
        print('Epoch #%d finished. *******************' % (epoch + 1))
        if (epoch + 1) % SNAPSHOT_FREQ == 0:
            save_results(SNAPSHOT_SAVE_DIR, MODEL_NAME, net, 
                         best_model, training_history, snapshot_idx = epoch)
            print('Snapshot saved.')
        
        
    print('Finished Training')
    time_elapsed = time.time() - start_time
    print('Completed in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed//3600, (time_elapsed % 3600)//60, time_elapsed % 60))
    
    save_results(MODEL_SAVE_DIR, MODEL_NAME, net,
                 best_model, training_history, snapshot_idx = -1)
    
    

