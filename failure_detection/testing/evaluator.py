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
from scipy import stats, ndimage
import scipy.misc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib 



class Evaluator():
    def __init__(self, nn_model, stride, patch_size, image_size, 
                patch_transform, device, classes, batch_size,
                net_input_size=224, particle_size=1,
                eval_bbox_size = 100):
        self.stride = stride
        self.patch_size = patch_size
        self.nn_model = nn_model
        self.im_height = image_size[0]
        self.im_width = image_size[1]
        self.patch_transform = patch_transform
        self.device = device
        self.classes = classes
        self.batch_size = batch_size
        self.net_input_size = net_input_size
        self.particle_size = particle_size
        self.eval_bbox_size = eval_bbox_size # size of the bounding box 
                                      # usef for eval in org image space
        
        
        # Dimensions of the processed image
        proc_img_height = floor((image_size[0] - patch_size) / stride) + 1
        proc_img_width = floor((image_size[1] - patch_size) /stride) + 1
        self.proc_img_height = proc_img_height
        self.proc_img_width = proc_img_width
        self.raw_image = []
        
        
        # proc_img holds the class probabilities for all queried patches
        # during processing the image.
        self.proc_img = np.zeros((len(classes), proc_img_height,proc_img_width), 
                                  dtype=float)
       
        # class_uncertainty holds the class uncertainty (variance) for each 
        # class.
        self.class_uncertainty = np.zeros(
                  (len(classes),proc_img_height,proc_img_width), dtype=float)
        
        self.mean_uncertainty = np.zeros(
                  (proc_img_height,proc_img_width), dtype=float)
        
        self.counter_proc_img = 0;
        self.patch_num = proc_img_height * proc_img_width
        self.total_patch_num = self.patch_num * self.particle_size
        print("Processing ", self.patch_num, " patches",
              " over each image.")
        print("Total evaluations evaluations per image",
              "(including the particles): ", self.total_patch_num)
        if self.batch_size > self.patch_num:
            print("The evaluation batch size is capped to the per image",
                  "patch number of", self.patch_num)
        

    
    # Loads preprocessed images (class probs and uncertainty values)
    def load_processed_image(self, image,  proc_img, mean_uncertainty):
        self.proc_img = proc_img
        self.mean_uncertainty = mean_uncertainty
        self.proc_img_pred = np.argmax(self.proc_img, 0)
        
        pil_img = transforms.ToPILImage()(image)
        pil_img = pil_img.convert(mode = "RGB")
        self.raw_image = pil_img
        
    def process_image(self, image):
        pil_img = transforms.ToPILImage()(image)
        
       
        pil_img = pil_img.convert(mode = "RGB")
        self.raw_image = pil_img
        
        input_patches_len = min(self.batch_size, self.patch_num)
        input_patches = torch.zeros([ self.particle_size * input_patches_len,
                              3, self.net_input_size, self.net_input_size],
                                    dtype=torch.float,
                                    device= self.device)
        output_array = torch.zeros(
              [self.particle_size* self.patch_num, 
                                    len(self.classes)],
                                   dtype=torch.float,
                                   device= self.device)
        
        softmax_layer = nn.Softmax(dim=1)
        # Turn on dropout during evaluation for uncertainty estimation
        if self.particle_size > 1:
            for mod in self.nn_model.modules():
                if mod.__class__.__name__.startswith('Dropout'):
                    mod.train()
        
        counter = 0
        batch_counter = 0 # counts the size of current batch (the last one will
                          # be of the same size)
        for i in range(self.proc_img_height):
            for j in range(self.proc_img_width):
                
                
                patch = transforms.functional.crop(pil_img,
                                                  i * self.stride, 
                                                  j * self.stride,
                                                  self.patch_size,
                                                  self.patch_size)

                patch_transformed = self.patch_transform(patch)
                patch_transformed = patch_transformed.to(self.device)
                patch_transformed = patch_transformed.reshape(
                                    1,
                                    patch_transformed.shape[0],
                                    patch_transformed.shape[1],
                                    patch_transformed.shape[2])
                
                start_idx = batch_counter * self.particle_size
                end_idx = (batch_counter + 1) * self.particle_size
                input_patches[start_idx:end_idx , :, :, :] =( 
                        patch_transformed.repeat(self.particle_size,1,1,1))
                
                if batch_counter == self.batch_size - 1:
                    with torch.set_grad_enabled(False):                       
                        # outputs shape: (patch_num * particle_num) * class_num
                        outputs = softmax_layer(self.nn_model(input_patches))
                        
                        start_idx = self.particle_size * (counter - 
                                                        batch_counter)
                        end_idx = self.particle_size * (counter + 1)
                        output_array[start_idx:end_idx, :]= outputs
                        batch_counter = -1
                # Take care of the last batch that does not reach its full size
                elif counter == self.patch_num - 1:
                    with torch.set_grad_enabled(False): 
                        latest_idx = (batch_counter + 1) * self.particle_size
                        
                        # outputs shape: (latest_idx) * class_num
                        outputs = softmax_layer(self.nn_model(
                                              input_patches[0:latest_idx]))
                        
                        start_idx = self.particle_size * (counter - 
                                                        batch_counter)
                        output_array[start_idx:, :]= outputs
                        batch_counter = -1
                    
                   
                counter+=1
                batch_counter+=1
        # Take the mean and variance of the class probabilities over the
        # particles
        # Reshaped to patch_num * particle_num * class_num
        output_array = output_array.reshape(self.patch_num,
                                            self.particle_size,
                                            len(self.classes))
        
        # class_num * patch_num
        class_prob = torch.mean(output_array, 1)
        class_var = torch.var(output_array, 1)
                
        # Reshape the output array to form an image with 4 channel. One for
        # the probability of each class
        class_prob = torch.transpose(class_prob, 0, 1)
        class_var = torch.transpose(class_var, 0, 1)
       
        class_prob = class_prob.reshape(len(self.classes), 
                                             self.proc_img_height,
                                             self.proc_img_width)
        class_var = class_var.reshape(len(self.classes), 
                                             self.proc_img_height,
                                             self.proc_img_width)
        self.proc_img = class_prob.cpu().numpy()
        self.proc_img_pred = np.argmax(self.proc_img, 0)
        self.class_uncertainty = class_var.cpu().numpy()
        self.mean_uncertainty = (torch.mean(class_var, 0)).cpu().numpy()
        
              
    def generate_heatmap(self, visualize=False, save=False, save_dir=None,
                         file_name=None, save_separate=False):
        cmap = 'viridis'
        interp = 'bilinear'
        
        self.counter_proc_img += 1
        
        if visualize:
            input("Press [enter] to continue.")
        
        plt.figure("main_fig")        
        plt.ion()
        
        if visualize:
            plt.show(block = False)
        
        plt.subplot(321)
        plt.title(self.classes[0])
        plt.imshow(self.proc_img[0, :, :],  cmap=cmap, interpolation=interp,
                   vmin=0.0, vmax=1.0)
        if self.counter_proc_img < 2:
            plt.colorbar()
     
        plt.subplot(322)
        plt.title(self.classes[1])
        plt.imshow(self.proc_img[1, :, :],  cmap=cmap, interpolation=interp,
                   vmin=0.0, vmax=1.0)
        if self.counter_proc_img < 2:
            plt.colorbar()
    
        plt.subplot(323)
        plt.title(self.classes[2])
        plt.imshow(self.proc_img[2, :, :],  cmap=cmap, interpolation=interp,
                   vmin=0.0, vmax=1.0)
        if self.counter_proc_img < 2:
            plt.colorbar()
        
        plt.subplot(324)
        plt.title(self.classes[3])
        plt.imshow(self.proc_img[3, :, :],  cmap=cmap, interpolation=interp,
                   vmin=0.0, vmax=1.0)
        if self.counter_proc_img < 2:
            plt.colorbar()
        
        plt.subplot(325)
        plt.title('Input Image')
        plt.imshow(self.raw_image)
        
        plt.subplot(326)
        plt.title('Prediction Uncertainty')
        plt.imshow(self.mean_uncertainty, cmap=cmap, 
                   interpolation=interp,
                   vmin=0.0, vmax=0.09)
        if self.counter_proc_img < 2:
            plt.colorbar()
            plt.tight_layout()
       
        
        if save:
            if (save_dir is not None) and (file_name is not None):
                fig_path = save_dir + '/' + file_name
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(fig_path, dpi=150)
            else:
                print("File path was not set for the generated heatmap.")
                exit()
         
        # If set to true, different heatmaps will be saved separately as well
        if save_separate:
            # Add up FP and FN to get failure probability
            fail_prob = self.proc_img[2, :, :] + self.proc_img[3, :, :]
            fig_folder = save_dir + '/failure_prob/'
            fig_path = save_dir + '/failure_prob/' + file_name
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            plt.figure()
            heatmap = plt.imshow(fail_prob,  cmap=cmap,interpolation=interp,
                   vmin=0.0, vmax=1.0)
            cbar = plt.colorbar()
            cbar.ax.set_title('failure prob.', rotation=0, pad=10)
            plt.tight_layout()
            plt.axis('off')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(9, 4)
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            
            # Save TP **********
            fig_folder = save_dir + '/true_pos/'
            fig_path = save_dir + '/true_pos/' + file_name
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            plt.figure()
            heatmap = plt.imshow(self.proc_img[0, :,:], 
                      cmap=cmap,interpolation=interp,
                      vmin=0.0, vmax=1.0)
            plt.title('True positive prob.')
            plt.axis('off')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(9, 4)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            # Save TN **********
            fig_folder = save_dir + '/true_neg/'
            fig_path = save_dir + '/true_neg/' + file_name
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            plt.figure()
            heatmap = plt.imshow(self.proc_img[1, :,:], 
                      cmap=cmap,interpolation=interp,
                      vmin=0.0, vmax=1.0)
            cbar = plt.colorbar()
            plt.title('True negative prob.')
            plt.axis('off')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(9, 4)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            # Save FP **********
            fig_folder = save_dir + '/false_pos/'
            fig_path = save_dir + '/false_pos/' + file_name
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            plt.figure()
            heatmap = plt.imshow(self.proc_img[2, :,:], 
                      cmap=cmap,interpolation=interp,
                      vmin=0.0, vmax=1.0)
            plt.title('False positive prob.')
            plt.axis('off')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(9, 4)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            # Save FN **********
            fig_folder = save_dir + '/false_neg/'
            fig_path = save_dir + '/false_neg/' + file_name
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            plt.figure()
            heatmap = plt.imshow(self.proc_img[3, :,:], 
                      cmap=cmap,interpolation=interp,
                      vmin=0.0, vmax=1.0)
            cbar = plt.colorbar()
            plt.title('False negative prob.')
            plt.axis('off')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(9, 4)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            # Save Uncertainty **********
            fig_folder = save_dir + '/uncertainty/'
            fig_path = save_dir + '/uncertainty/' + file_name
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            plt.figure()
            plt.imshow(self.mean_uncertainty, cmap=cmap, interpolation=interp,
                       vmin=0.0, vmax=0.12)
            cbar = plt.colorbar()
            plt.title('Uncertainty score')
            plt.axis('off')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(9, 4)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
    # This function saves the class probabilites and uncertainty values 
    # for the processed image. The data is essentially the same heatmaps
    # that generate_heatmap() visualizes. But this function saves the result
    # of each class separately for post processing purposes
    def save_processed_images(self, save_dir, file_name):          
        for i in range(len(self.classes)):
            path = save_dir + '/' +self.classes[i]
            outfile = path + '/' + file_name + '.txt'
            
            if not os.path.exists(path):
                os.makedirs(path)
            np.savetxt(outfile, self.proc_img[i, :, :])
        
        path = save_dir + '/uncertainty/'
        outfile = path + file_name + '.txt'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(outfile, self.mean_uncertainty)

        
        # Save the metadata (patch size and stride). There is supposed to
        # be one metadata file per bagfile. Right now I am naively rewriting
        # this for each image
        info = {'stride': self.stride,
                'patch_size': self.patch_size}
        info_json = json.dumps(info)
        file = open(save_dir + '/' + 'meta_data.json', 'w');
        file.write(info_json)
        file.close()
        
      
                
    # Converts the input pixel coordinate from the original image space to
    # the corresponding bounding box (coordinates of the diagonal 
    # corners) in the reduced processed image (after being evaluated by the NN 
    # on overlapping patches)
    def get_bbox_in_proc_img(self, points):
        bbox_half_size = floor((self.eval_bbox_size / float(self.stride)) / 2.0)
        points_conv = np.round((points - (self.patch_size/2.0)).astype(float) / 
                               self.stride)
        points_conv = points_conv.astype(int)
        corner_pts1 = points_conv - bbox_half_size
        corner_pts2 = points_conv + bbox_half_size
        

        return corner_pts1, corner_pts2
    
    # Runs post processing on the processed image for a given list of 
    # query points to predict their corresponding class
    # query_points: np.array(n*2)
    def predict(self, query_points):
        predictions = np.zeros((query_points.shape[0]), dtype=int)
        cor1, cor2 = self.get_bbox_in_proc_img(query_points)
        
        # Take the majority vote in the bounding box as prediction
        for i in range(query_points.shape[0]):
            neighbor_preds = self.proc_img_pred[cor1[i,0] : cor2[i,0]+1,
                                                cor1[i,1] : cor2[i,1]+1]
           
            
            #******************
            # KNN
            mode = stats.mode(neighbor_preds, axis=None)
            predictions[i] = mode[0]
 
            
        return predictions
      
    # Testing different smoothing methods  
    def predict_filtered(self, query_points):
        predictions = np.zeros((query_points.shape[0]), dtype=int)
        cor1, cor2 = self.get_bbox_in_proc_img(query_points)
     
        
        #******************
        # Mean Filter
        kernel = np.ones((3,3), dtype=float) / 9.0
        filter_thresh = 0.3
        
        false_neg = self.proc_img_pred == 3
        false_neg = np.asarray(false_neg, dtype=float)
        smoothed_fn = ndimage.filters.convolve(false_neg, kernel)
        smoothed_fn_bool = smoothed_fn > filter_thresh
        self.proc_img_pred[smoothed_fn_bool] = 3
        
        # Take the majority vote in the bounding box as prediction
        for i in range(query_points.shape[0]):
            neighbor_preds = self.proc_img_pred[cor1[i,0] : cor2[i,0]+1,
                                                cor1[i,1] : cor2[i,1]+1]
           
            
            #******************
            # KNN
            mode = stats.mode(neighbor_preds, axis=None)
            predictions[i] = mode[0]

            
        return predictions
      
