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

# This torch dataset is for the purpose of loading full images as well as
# initial evaluation results (processed images). This dataset is supposed
# to be used for post processing (pp) the netwrok output.

import os
import torch
import json
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class FailureDetectionDataset(Dataset):
    
    def __init__(self, root_dir, eval_results_dir, bagfile_list,
                 loaded_image_channel=1, output_image_channel=1,
                 transform=None, meta_data_dir = None,
                 read_class_label_from_file=True,
                 load_patch_info=False,
                 segment_id = None, segment_length=None):
        DATA_FILE = "full_images_data.json"
        NAMES_FILE = "full_images_names.json"
        PATCH_DATA_FILE = "image_patches_data.json"
        PROC_IMG_META_DATA_FILE = "meta_data.json"
        self.PROC_IMG_FOLDERS = ['perception_true_pos',
                                  'perception_true_neg',
                                  'perception_false_pos',
                                  'perception_false_neg']
        self.PROC_IMG_UNCERTAINTY_FOLDER = 'uncertainty'
        
        
        self.root_dir = root_dir
        self.eval_results_dir = eval_results_dir
        self.meta_data_dir = meta_data_dir
        if not (meta_data_dir):
            meta_data_dir = root_dir
        self.meta_data_dir = meta_data_dir
        self.loaded_image_channel = loaded_image_channel
        self.output_image_channel = output_image_channel
        self.transform = transform
        self.load_patch_info = load_patch_info
        self.segment_id = segment_id
        self.segment_length = segment_length
       
        self.corr_img_patch_indices = []
        self.full_img_names = []
        self.jpp_obs_existence = []
        self.kinect_obs_existence = []
        self.patch_coord_left_x = []
        self.patch_coord_left_y = []
        self.multi_class_labels = []
        self.proc_img_meta_data = {}
        
        # Check if the data loader should operate in segmentation mode
        # Segmentation mode allows for processing a specific part of a bagfile 
        # This mode is not supported if more than one bagfiles are asked
        # to be loaded
        self.seg_mode = False
        if (segment_id is not None) and (segment_length is not None):
            if len(bagfile_list) == 1:
                self.seg_mode = True
                print("Dataloader in segmentation mode: bagfile#", 
                      bagfile_list[0], "segment#: ", segment_id,
                      "segment length: ", segment_length)
            else:
                self.seg_mode = False
                print("Dataloader cannot run in segmentation mode because "
                      "more than one bagfiles are requested.")
                exit()
                
        
        # loaded image type of "mono" and output image type of "RGB" is not
        # supported
        if (loaded_image_channel == 1) and (output_image_channel == 3):
            print("Input image mode of mono and output of RGB is not supported")
            exit()
         
        patch_id_offset = 0 
        for i in bagfile_list:
            bagfile_num = i
            bagfile_dir = "{0:05d}".format(bagfile_num)
            bagfile_path = root_dir + '/' + bagfile_dir + '/'
            bagfile_path_meta_data = meta_data_dir + '/' + bagfile_dir + '/'

            with open(bagfile_path_meta_data + NAMES_FILE, 'r') as file:
                names_obj = json.load(file)
       
            self.full_img_names = (self.full_img_names +
                                names_obj["img_names_left"]);
            
            if load_patch_info:
                for img_info in names_obj["corr_img_patch_indices"]:
                    patch_idx_np = np.array(img_info["indices"])
                    patch_idx_np = patch_idx_np + patch_id_offset
                    corrected_patch_idx = patch_idx_np.tolist()
                    self.corr_img_patch_indices = (
                        self.corr_img_patch_indices + [corrected_patch_idx])
                    
                    
            # Load the processed images' meta data (stride and patch size)
            proc_meta_path = (eval_results_dir + '/processed_imgs/' +
                              bagfile_dir + '/' + PROC_IMG_META_DATA_FILE)
            with open(proc_meta_path, 'r') as file:
                proc_img_metadata = json.load(file)
            self.proc_img_meta_data[bagfile_dir] = proc_img_metadata
               
            
            if load_patch_info:
                file_path = bagfile_path_meta_data + PATCH_DATA_FILE 
                with open(file_path, 'r') as file:
                    patch_data_obj = json.load(file)
                    
                self.jpp_obs_existence = (self.jpp_obs_existence +
                                  patch_data_obj['jpp_obs_existence'])
                self.kinect_obs_existence = (self.kinect_obs_existence +
                                  patch_data_obj['kinect_obs_existence'])
                self.patch_coord_left_x = (self.patch_coord_left_x +
                                  patch_data_obj["patch_coordinate_left"]["x"])
                self.patch_coord_left_y = (self.patch_coord_left_y +
                                  patch_data_obj["patch_coordinate_left"]["y"])
                if read_class_label_from_file:
                    self.multi_class_labels = (self.multi_class_labels +
                               patch_data_obj['multi_class_label'])
                patch_id_offset += len(patch_data_obj['jpp_obs_existence'])
            
        if self.load_patch_info:
            self.jpp_obs_existence = np.asarray(self.jpp_obs_existence)
            self.kinect_obs_existence = np.asarray(self.kinect_obs_existence)
            if read_class_label_from_file:
                self.multi_class_labels = np.asarray(self.multi_class_labels)
            
            # Prints dataset statistics:
            # True positive
            if read_class_label_from_file:
                true_pos = self.multi_class_labels == 0
            else:
                true_pos = ((self.jpp_obs_existence ==
                             self.kinect_obs_existence) &
                            self.kinect_obs_existence)

            # True negative
            if read_class_label_from_file:
                true_neg = self.multi_class_labels == 1
            else:
                true_neg = ((self.jpp_obs_existence ==
                             self.kinect_obs_existence) &
                            np.logical_not(self.kinect_obs_existence))

            # False positive
            if read_class_label_from_file:
                false_pos = self.multi_class_labels == 2
            else:
                false_pos = ((self.jpp_obs_existence !=
                              self.kinect_obs_existence) &
                             np.logical_not(self.kinect_obs_existence))

            # False negative
            if read_class_label_from_file:
                false_neg = self.multi_class_labels == 3
            else:
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
            self.class_label = ['true_pos', 'true_neg', 'false_pos', 
                                'false_neg']
            class_weights = (float(self.kinect_obs_existence.size) / 
                            self.class_count)
            
            print('class weights: ', class_weights)
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
            
            # Calculate the offset index (starting idx) for the current segment
            # if in segmentation mode. Also check if the requested segment
            # exists within the length of the bagfile 
            if self.seg_mode:
                bagfile_size = len(self.full_img_names)
                self.seg_offset = self.segment_id * self.segment_length
                if self.seg_offset >= bagfile_size:
                    print("Segment#", self.segment_id, "does not exist "
                          "for bagfile#", bagfile_list[0],
                          self.seg_offset, ">=", (bagfile_size))
                    exit()
                # The actual length of the segment could be smaller than the 
                # requested segment length(for the last segment)
                self.seg_actual_len = min(bagfile_size - self.seg_offset,
                                          self.segment_length)
                
              
        
    def __len__(self):
        if self.seg_mode:
            return self.seg_actual_len
        else:
            return len(self.full_img_names)
       
    def __getitem__(self, idx):
        # Offset the id if in segmentation mode:
        if self.seg_mode:
            idx_off = idx + self.seg_offset
        else:
            idx_off = idx
            
        curr_full_image_name = self.full_img_names[idx_off]
        curr_bagfile_prefix = curr_full_image_name[0:5];
        full_image_path = self.root_dir+ '/' + curr_bagfile_prefix + '/' \
                     + 'images/left_cam/' + curr_full_image_name
        full_img = io.imread(full_image_path)
      
        full_img = full_img.reshape((full_img.shape[0], full_img.shape[1],
                                     self.loaded_image_channel))
        
        # Transform the images to PIL image
        full_img = transforms.ToPILImage()(full_img)
        if self.output_image_channel != self.loaded_image_channel:
            if self.output_image_channel == 1:
                full_img = full_img.convert(mode = "L")
        if self.transform:
            full_img = self.transform(full_img)
            
        # Load the evaluation resutls (class probability and uncertainty
        # heatmaps)
        all_proc_imgs = np.zeros([])
        for i in range(len(self.PROC_IMG_FOLDERS)):
            file_path = (self.eval_results_dir + '/processed_imgs/' +
                    curr_bagfile_prefix + '/' + self.PROC_IMG_FOLDERS[i] +
                    '/' + curr_full_image_name[:-4] + '.txt')
            next_proc_img = np.loadtxt(file_path, dtype=float)
            next_proc_img = next_proc_img.reshape((1, next_proc_img.shape[0],
                                                  next_proc_img.shape[1]))
            if i == 0:
                all_proc_imgs = np.zeros((4, next_proc_img.shape[1],
                                             next_proc_img.shape[2]), 
                                         dtype=float)
            all_proc_imgs[i, :, :] = next_proc_img
        
        file_path = (self.eval_results_dir + '/processed_imgs/' +
                curr_bagfile_prefix + '/' + self.PROC_IMG_UNCERTAINTY_FOLDER +
                '/' + curr_full_image_name[:-4] + '.txt')       
        uncertainty_vals = np.loadtxt(file_path, dtype=float)
        
        proc_img_meta_data = self.proc_img_meta_data[curr_bagfile_prefix]
       
        if self.load_patch_info:
            patch_coord_l = np.array([], dtype=int).reshape(0, 2)
            jpp_obs_existence = []
            kinect_obs_existence = []
            multi_class_label = []
            for j in self.corr_img_patch_indices[idx_off]:
                # Prepare patch_coords_left for current full_img
                new_instance = np.array([self.patch_coord_left_y[j],
                                  self.patch_coord_left_x[j]]).reshape(1,2)
                patch_coord_l = np.append(patch_coord_l, new_instance, 0)
                
                # Prepare jpp_obs_existence for current full_img
                jpp_obs_existence = (jpp_obs_existence + 
                                    [self.jpp_obs_existence[j]])
               
                # Prepare kinect_obs_existence for current full_img
                kinect_obs_existence = (kinect_obs_existence + 
                                    [self.kinect_obs_existence[j]])
                
                # Prepare multi_class_label for current full_img
                multi_class_label = (multi_class_label + 
                                    [self.multi_class_labels[j]])
          
            sample = {'full_img': full_img,
                      'full_img_name': curr_full_image_name,
                      'bagfile_name': curr_bagfile_prefix,
                      'proc_imgs': all_proc_imgs,
                      'proc_img_uncertainty': uncertainty_vals,
                      'proc_img_meta_data': proc_img_meta_data,
                      'patch_coords_left':patch_coord_l,
                      'jpp_obs_existence': jpp_obs_existence,
                      'kinect_obs_existence': kinect_obs_existence,
                      'multi_class_label':  multi_class_label}
          
        else:
            sample = {'full_img': full_img, 
                      'full_img_name': curr_full_image_name,
                      'bagfile_name': curr_bagfile_prefix}
       
        return sample
  
