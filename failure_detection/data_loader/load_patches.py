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

import os
import torch
import json
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class FailureDetectionDataset(Dataset):
    
    def __init__(self, root_dir, bagfile_list, 
                 loaded_image_channel=1, output_image_channel=1, 
                 transform=None, meta_data_dir = None,
                 extract_from_full_img = False,
                 patch_size = 100,
                 class_weights_coeff = [1.0, 1.0, 1.0, 1.0]):
        _PATCH_DATA_FILE = "image_patches_data.json"
        _PATCH_NAME_FILE = "image_patches_names.json"
        
        # root directory is the dataset directory with the images and 
        # meta_data_dir is the dataset directory with the same architecture,
        # but only the meta data will be read from there (the JSON files)
        # The reason for separating the two is that different perception 
        # models can use one common image dataset and their own metadata 
        # (labels)
        self.root_dir = root_dir
        self.meta_data_dir = meta_data_dir
        if not (meta_data_dir):
            meta_data_dir = root_dir
            
        self.meta_data_dir = meta_data_dir
        self.loaded_image_channel = loaded_image_channel
        self.output_image_channel = output_image_channel
        self.transform = transform
        self.extract_from_full_img = extract_from_full_img
        self.patch_size = patch_size
        self.patch_names = []
        self.full_img_names = []
        self.jpp_obs_existence = []
        self.kinect_obs_existence = []
        self.patch_coord_left_x = []
        self.patch_coord_left_y = []
        
        # loaded image type of "mono" and output image type of "RGB" is not
        # supported
        if (loaded_image_channel == 1) and (output_image_channel == 3):
            print("Input image mode of mono and output of RGB is not supported")
            exit()
            
        if (root_dir != meta_data_dir) and (not(extract_from_full_img)):
            print("Warning: Patches are asked to be loaded from file although"
                  " the data directory and meta data directory are not the"
                  " same!")
        if (root_dir == meta_data_dir) and (extract_from_full_img):
            print("Notice: Patches are asked to be extracted from full images" 
                  " although the data directory and meta data directory"
                  " are the same. This is inefficient if extracted patches"
                  " already exist on disk.")
         
        for i in bagfile_list:
            bagfile_num = i
            bagfile_dir = "{0:05d}".format(bagfile_num)
            bagfile_path = root_dir + '/' + bagfile_dir + '/'
            bagfile_path_meta_data = meta_data_dir + '/' + bagfile_dir + '/'
            
            with open(bagfile_path_meta_data + _PATCH_NAME_FILE, 'r') as file:
                patch_name_obj = json.load(file)
        
            self.patch_names = (self.patch_names +
                                patch_name_obj["img_patch_name"]);
            self.full_img_names = (self.full_img_names +
                                patch_name_obj["corr_full_img_name"]);
            
            with open(bagfile_path_meta_data + _PATCH_DATA_FILE, 'r') as file:
                patch_data_obj = json.load(file)

            self.jpp_obs_existence = (self.jpp_obs_existence +
                                      patch_data_obj['jpp_obs_existence'])
            self.kinect_obs_existence = (self.kinect_obs_existence +
                                      patch_data_obj['kinect_obs_existence'])
            self.patch_coord_left_x = (self.patch_coord_left_x +
                                   patch_data_obj["patch_coordinate_left"]["x"])
            self.patch_coord_left_y = (self.patch_coord_left_y +
                                   patch_data_obj["patch_coordinate_left"]["y"])
            
            
        self.jpp_obs_existence = np.asarray(self.jpp_obs_existence)
        self.kinect_obs_existence = np.asarray(self.kinect_obs_existence)
        
        # Prints dataset statistics:
        # True positive
        true_pos = ((self.jpp_obs_existence == 
                            self.kinect_obs_existence) & 
                            self.kinect_obs_existence)
        
        # True negative
        true_neg = ((self.jpp_obs_existence == 
                            self.kinect_obs_existence) & 
                            np.logical_not(self.kinect_obs_existence))
        
        # False positive
        false_pos = ((self.jpp_obs_existence != 
                            self.kinect_obs_existence) & 
                            np.logical_not(self.kinect_obs_existence))
        
        # False negative
        false_neg = ((self.jpp_obs_existence != 
                            self.kinect_obs_existence) & 
                            self.kinect_obs_existence)
        
        print('True Pos: ', true_pos.sum(), ', ', true_pos.sum() / 
                        true_pos.size)
        print('True Neg: ', true_neg.sum(), ', ', true_neg.sum() / 
                        true_neg.size)
        print('False Pos: ', false_pos.sum(), ', ', false_pos.sum() / 
                        false_pos.size)
        print('False Neg: ', false_neg.sum(), ', ', false_neg.sum() / 
                        false_neg.size)
        
        self.class_count = np.array([true_pos.sum(), true_neg.sum(),
                                     false_pos.sum(), false_neg.sum()])
        self.class_label = ['true_pos', 'true_neg', 'false_pos', 'false_neg']
        class_weights = (float(self.kinect_obs_existence.size) / 
                         self.class_count)
        
        print('balanced class weights: ', class_weights)
        for k in range(4):
            class_weights[k] = class_weights[k] * class_weights_coeff[k]
        print('Adjusted class weights: ', class_weights)
        sample_weights = np.zeros(self.kinect_obs_existence.size)
        sample_weights[true_pos] = class_weights[0]
        sample_weights[true_neg] = class_weights[1]
        sample_weights[false_pos] = class_weights[2]
        sample_weights[false_neg] = class_weights[3]
        self.sample_weights = sample_weights
        
        self.jpp_obs_existence = self.jpp_obs_existence.tolist()
        self.kinect_obs_existence = self.kinect_obs_existence.tolist()
                       
        true_pos = np.reshape(true_pos ,(-1, 1))
        true_neg = np.reshape(true_neg, (-1, 1))
        false_pos = np.reshape(false_pos, (-1, 1))
        false_neg = np.reshape(false_neg, (-1, 1))
        all_classes = np.concatenate((true_pos, true_neg, false_pos, 
                                      false_neg), axis=1)

        self.multi_class_labels = np.argmax(all_classes, axis=1)
        
    def __len__(self):
        return len(self.patch_names)
       
    def __getitem__(self, idx):
        curr_patch_name = self.patch_names[idx]
        curr_full_image_name = self.full_img_names[idx]
        curr_bagfile_prefix = curr_patch_name[0:5];
        patch_path = self.root_dir + '/' + curr_bagfile_prefix + '/' \
                     + 'images/left_cam_patch/' + curr_patch_name
        full_image_path = self.root_dir + '/' + curr_bagfile_prefix + '/' \
                     + 'images/left_cam/' + curr_full_image_name
                  
       
        patch_coord_left = np.array([self.patch_coord_left_x[idx],
                                     self.patch_coord_left_y[idx]])
        jpp_obs_existence = self.jpp_obs_existence[idx]
        kinect_obs_existence = self.kinect_obs_existence[idx]
        #binary_label = jpp_obs_existence == kinect_obs_existence
        if (jpp_obs_existence == kinect_obs_existence):
            binary_label = 1
        else:
            binary_label = 0
            
        if self.extract_from_full_img:
            full_img = io.imread(full_image_path)
            full_img = full_img.reshape((full_img.shape[0], full_img.shape[1], 
                               self.loaded_image_channel))
            half_size = floor(self.patch_size / 2)
            patch = full_img[patch_coord_left[1] - half_size:
                             patch_coord_left[1] + half_size,
                             patch_coord_left[0] - half_size:
                             patch_coord_left[0] + half_size, :]
           
        else:
            patch = io.imread(patch_path)
            patch = patch.reshape((patch.shape[0], patch.shape[1], 
                        self.loaded_image_channel))
       
        
        # Transform the images to PIL image
        patch = transforms.ToPILImage()(patch)
       
        
        if self.output_image_channel != self.loaded_image_channel:
            if self.output_image_channel == 1:
                patch = patch.convert(mode = "L")
       
        
        if self.transform:
            patch = self.transform(patch)
        
        sample = {'patch_name': curr_patch_name, 'patch': patch, 
                  #'full_img': full_img,
                  'patch_coord_left': patch_coord_left,
                  'jpp_obs_existence': jpp_obs_existence,
                  'kinect_obs_existence': kinect_obs_existence,
                  'binary_label': binary_label,
                  'multi_class_label':  self.multi_class_labels[idx]}
       
        return sample
    
    def get_dataset_info(self):
        return {'class_label': self.class_label,
                'class_count': self.class_count,
                'sample_weights': self.sample_weights}
        

    
    
