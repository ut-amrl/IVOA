import os
from os.path import isfile, join
import torch
import json
import csv
import re
import logging
from skimage import io, transform
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# TODO: add a check for regression mode not being supported 


# This dataset loads full images and the corresponding error label images
class DepthErrorDataset(Dataset):
  def __init__(self, root_dir, raw_img_dir, session_list,
               root_refined_model_dir=None,
               loaded_image_color=False, 
               output_image_color=False,
               session_prefix_length=5,
               raw_img_folder="image_0",
               label_img_folder="depth_err_labels_abs1.0_rel_0.2_maxRange30_ds",
               mask_img_folder="valid_pixel_mask_abs1.0_rel_0.2_maxRange30_ds",
               no_meta_data_available=False,
               load_only_with_labels=False,
               transform_input=None,
               transform_target=None,
               load_masks=False,
               regression_mode=False,
               binarize_target=True,
               stereo_mode=False,
               raw_img_folder_second_camera=None):
    DESCRIPTOR_FILE = "descriptors.csv"
    IMG_NAMES_FILE = "img_names.json"
    KEYPOINTS_FILE = "keypoints.json"
    
    self.binarize_target = binarize_target
    # The threshold that is used to binarize images with pixel values
    # between 0 and 255.
    self.binarize_thresh = 10
    self.root_dir = root_dir
    self.root_refined_model_dir = root_refined_model_dir
    self.raw_img_dir = raw_img_dir
    self.loaded_image_color=loaded_image_color
    self.output_image_color=output_image_color
    self.transform_input = transform_input
    self.transform_target = transform_target
    self.session_name_format = ""
    self.raw_img_folder = raw_img_folder # "image_0", "img_left"
    self.raw_img_folder_second_camera = raw_img_folder_second_camera
    self.label_img_folder = label_img_folder
    self.mask_img_folder = mask_img_folder
    # If set to true, it means that image file name is NOT available. 
    # The data loader should look at the raw_img_dir, and load all 
    # available images if training labels are not explicitly requested.
    # If load_only_with_labels==True, then the list of images should
    # be created from the folder of training label files.
    self.no_meta_data_available=no_meta_data_available

    # If set to true loads only datapoints for which a training label
    # is available. The list of images to be loaded is by default loaded
    # from the meta data files but if no_meta_data_available then the list
    # is created by actually reading through available label files.
    self.load_only_with_labels=load_only_with_labels
    self.load_masks=load_masks
    self.regression_mode=regression_mode
    self.stereo_mode=stereo_mode
    
    if no_meta_data_available and (not load_only_with_labels):
      self.load_labels=False
    else:
      self.load_labels=True

    if self.root_refined_model_dir is not None and self.load_labels:
      self.load_refined_model_labels=True
      print("Loading refined model labels from ", self.root_refined_model_dir)
    else:
      self.load_refined_model_labels=False

    # The length of the session name prefix
    if (session_prefix_length == 2):
      self.session_name_format = "{0:02d}"
    elif (session_prefix_length == 5):
      self.session_name_format = "{0:05d}"
    else:
      logging.error("Session prefix length of %d is not supported.",
                    session_prefix_length)
      exit()
      
    if ((not loaded_image_color) and output_image_color):
      logging.error("Cannot convert from mono to color.")
      exit()
      
    if ((raw_img_folder_second_camera is None) and stereo_mode):
      logging.error("Running in stereo mode but no second camera image folder specified.")
      exit()
    
    self.img_names = []
    # List of corresponding session number for all image names
    self.session_num = np.array([], dtype=int)

    for session_num in session_list:
      session_folder = self.session_name_format.format(session_num)
      session_path = root_dir + '/' + session_folder 

      if self.no_meta_data_available:
        if self.load_labels:
          img_dir = os.path.join(self.root_dir, session_folder,self.label_img_folder)
        else:
          img_dir = (self.raw_img_dir + '/' + session_folder +
                   '/' + self.raw_img_folder)
        session_img_names = self.get_img_names(img_dir)
        self.img_names = self.img_names + session_img_names
      else:
        with open(session_path + IMG_NAMES_FILE, 'r') as file:
          img_names_file_obj = json.load(file)
        session_img_names = img_names_file_obj["img_name"]
        self.img_names = self.img_names + session_img_names

      session_num_list = (session_num * 
                  np.ones(len(session_img_names))).astype(int)
      self.session_num = np.concatenate((self.session_num, session_num_list))

  def __len__(self):
    return len(self.img_names)

  # Returns the raw image as well as the corresponding heatmap image that 
  # works as a metric of which parts of the input image are bad for VO/SLAM
  def __getitem__(self, idx):
    session_folder = self.session_name_format.format(self.session_num[idx])
             
    curr_img_path = os.path.join(self.raw_img_dir, session_folder, self.raw_img_folder, self.img_names[idx].rstrip())                    
    curr_img_label_path = os.path.join(self.root_dir, session_folder,self.label_img_folder, self.img_names[idx].rstrip())
    curr_img_mask_path = os.path.join(self.root_dir, session_folder,self.mask_img_folder, self.img_names[idx].rstrip())                       
    
    if self.load_refined_model_labels:
      curr_img_label_refined_model_path = os.path.join(self.root_refined_model_dir, session_folder,self.label_img_folder, self.img_names[idx].rstrip())
    
    if self.stereo_mode:
      curr_img_sec_path = os.path.join(self.raw_img_dir, session_folder, self.raw_img_folder_second_camera, self.img_names[idx].rstrip()) 
    
  

    # *****************
    # Load target image if available
    labels=torch.empty(0)
    if self.load_labels:
      label_img = io.imread(curr_img_label_path)
      label_img = label_img.reshape((label_img.shape[0], label_img.shape[1], 1))
      label_img_width = label_img.shape[1]
      label_img_height = label_img.shape[0]
      if self.binarize_target:
        bin_msk = label_img > self.binarize_thresh
        label_img[bin_msk] = 255
        label_img[np.logical_not(bin_msk)] = 0

      # Transform the images to PIL image
      label_img = transforms.ToPILImage()(label_img)

      if self.transform_target:
        label_img = self.transform_target(label_img)

      # Create binary class labels if not in regression mode
      if not self.regression_mode:
        if not torch.is_tensor(label_img):
          label_img = transforms.functional.to_tensor(label_img)
        labels = torch.zeros_like(label_img, dtype=torch.int64)
        bin_msk = label_img > (self.binarize_thresh / 255.0)
        labels[bin_msk] = 1
        
    if self.load_labels and self.load_refined_model_labels:
      # Load labels corresponding to the prediction errors of the refined model
      label_img_ref = io.imread(curr_img_label_refined_model_path)
      label_img_ref = label_img_ref.reshape((label_img_ref.shape[0], label_img_ref.shape[1], 1))
      label_img_width_ref = label_img_ref.shape[1]
      label_img_height_ref = label_img_ref.shape[0]
      if self.binarize_target:
        bin_msk = label_img_ref > self.binarize_thresh
        label_img_ref[bin_msk] = 255
        label_img_ref[np.logical_not(bin_msk)] = 0

      # Transform the images to PIL image
      label_img_ref = transforms.ToPILImage()(label_img_ref)

      if self.transform_target:
        label_img_ref = self.transform_target(label_img_ref)

      # Create binary class labels if not in regression mode
      if not self.regression_mode:
        if not torch.is_tensor(label_img_ref):
          label_img_ref = transforms.functional.to_tensor(label_img_ref)
        labels_ref = torch.zeros_like(label_img_ref, dtype=torch.int64)
        bin_msk = label_img_ref > (self.binarize_thresh / 255.0)
        labels_ref[bin_msk] = 1
        

    # Load the target image masks
    if self.load_labels and self.load_masks:
      mask_img = io.imread(curr_img_mask_path)
      mask_img = mask_img.reshape((mask_img.shape[0], mask_img.shape[1], 1))
      mask_img_width = mask_img.shape[1]
      mask_img_height = mask_img.shape[0]

      if ((mask_img_width != label_img_width) or 
          (mask_img_height != label_img_height)):
        logging.error("Mask image and label image sizes are not equal")
        exit()

      # Transform the images to PIL image
      mask_img = transforms.ToPILImage()(mask_img)

      if self.transform_target:
        mask_img = self.transform_target(mask_img)

      # In classification/segmentation mode, the mask is applied to the labels.
      # masked pixels will get a label class of -1
      if not self.regression_mode:
        if not torch.is_tensor(mask_img):
          mask_img = transforms.functional.to_tensor(mask_img)
        mask_img = mask_img > (self.binarize_thresh / 255.0)
        labels[torch.logical_not(mask_img)] = -1
        
      if labels.nelement() != 0:
        labels = torch.reshape(labels, (label_img.size()[1], 
                                        label_img.size()[2])) 
        

      if self.load_refined_model_labels:
        if ((mask_img_width != label_img_width_ref) or 
            (mask_img_height != label_img_height_ref)):
          logging.error("Mask image and label image sizes are not equal")
          exit()
        
        # In classification/segmentation mode, the mask is applied to the labels.
        # masked pixels will get a label class of -1. Apply this to labels_ref
        if not self.regression_mode:
          labels_ref[torch.logical_not(mask_img)] = -1
          
        if labels_ref.nelement() != 0:
          labels_ref = torch.reshape(labels_ref, (label_img_ref.size()[1], 
                                          label_img_ref.size()[2])) 
        


    # *****************
    # Load input image
    img = io.imread(curr_img_path)

    # Handle color and mono images accordingly
    raw_img_channels = 1
    if self.loaded_image_color:
      raw_img_channels = 3
      img = img[:, : , 0:raw_img_channels]

    img = img.reshape((img.shape[0], img.shape[1], raw_img_channels))

    # Transform the images to PIL image
    img = transforms.ToPILImage()(img)

    # Convert the input image to mono if required by the arguments
    if self.loaded_image_color != self.output_image_color:
      if not self.output_image_color:
        img = img.convert(mode="L")

    if self.transform_input:
      img = self.transform_input(img)

    # *****************
    # Load the secondary image if in stereo mode and concatenate the input images
    if self.stereo_mode:
      img_sec = io.imread(curr_img_sec_path)

      # Handle color and mono images accordingly
      raw_img_channels = 1
      if self.loaded_image_color:
        raw_img_channels = 3
        img_sec = img_sec[:, : , 0:raw_img_channels]

      img_sec = img_sec.reshape((img_sec.shape[0], img_sec.shape[1], raw_img_channels))

      # Transform the images to PIL image
      img_sec = transforms.ToPILImage()(img_sec)

      # Convert the input image to mono if required by the arguments
      if self.loaded_image_color != self.output_image_color:
        if not self.output_image_color:
          img_sec = img_sec.convert(mode="L")

      if self.transform_input:
        img_sec = self.transform_input(img_sec)
        
      # Check if img is a tensor or PIL image and stack them accordingly
      if not torch.is_tensor(img_sec):
        img = np.concatenate((img, img_sec), axis=0)
      else:
        img = torch.cat((img, img_sec), axis=0)
      

    if not self.load_labels:
      sample = {'img': img,
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}
    elif self.load_masks:
      if self.load_refined_model_labels:
        sample = {'img': img,
                'labels': labels,
                'labels_refined_model': labels_ref,
                'mask_img' : mask_img,
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}        
      else:
      sample = {'img': img,
                'labels': labels,
                'mask_img' : mask_img,
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}
    else:
      if self.load_refined_model_labels:
        sample = {'img': img,
                'labels': labels,
                'labels_refined_model': labels_ref, 
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}
      else:        
      sample = {'img': img,
                'labels': labels, 
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}


    return sample
  
  def get_dataset_info(self):
    return {'size': len(self.img_names)}

  # Returns a list of name of all images in the input directory
  def get_img_names(self, dir):
    if os.path.isdir(dir):
      img_names = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    else:
      logging.error("Directory does not exist: ", dir)
      exit()
    return img_names

if __name__ == "__main__":
  data_transform_input = transforms.Compose([
            transforms.Resize((300, 480)),
            transforms.ToTensor()
  ])

  root_dir ='/robodata/user_data/srabiee/results/ipr/depth_prediction/ganet_deep_airsim_01_dropout/model_ensemble_epoch_14_e012_tmp/cityenv_wb/'
  raw_img_dir = '/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/'
  sessions = [1007]
  dataset = DepthErrorDataset(root_dir,
                                raw_img_dir,
                                sessions,
                                loaded_image_color=True,
                                output_image_color=True,
                                session_prefix_length=5,
                                raw_img_folder="img_left",
                                label_img_folder="depth_err_labels_abs1.0_rel_0.2_maxRange30_ds",
                                no_meta_data_available=True,
                                load_only_with_labels=True,
                                transform_input=data_transform_input,
                                transform_target=data_transform_input,
                                load_masks=True,
                                regression_mode=False,
                                binarize_target=True)
  print("dataset size: ", dataset.__len__())                                
  sample = dataset[0]
