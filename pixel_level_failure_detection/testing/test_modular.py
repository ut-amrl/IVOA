import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import time
import argparse, os
import shutil
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from distutils.version import LooseVersion
from pixel_level_failure_detection.data_loader.load_images import DepthErrorDataset
import csv

# Third party libs
from pixel_level_failure_detection.config import cfg
from networks.models import ModelBuilder, SegmentationModule
from lib.nn import patch_replication_callback
from lib.utils.utils import parse_devices
from pixel_level_failure_detection.lib.utils.utils import MaskedMSELoss

from depth_utilities import colorize

# TODO: 
# 1- update the dataset and use the additional config params

# This script is used for running network architecture trained in
# train_modular.py on test data

def group_weight(module):
  group_decay = []
  group_no_decay = []
  for m in module.modules():
    if isinstance(m, nn.Linear):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, nn.modules.conv._ConvNd):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
      if m.weight is not None:
        group_no_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)

  assert len(list(module.parameters())) == len(group_decay) + len(
    group_no_decay)
  groups = [dict(params=group_decay),
            dict(params=group_no_decay, weight_decay=.0)]
  return groups

def compute_classification_report_from_cnf(cnf_matrix):
  """
  Computes the classification report from the confusion matrix.
  :param cnf_matrix: confusion matrix
  :return: classification report
  """
  report = []
  # Compute the classification report
  sum_precision = 0.0
  sum_recall = 0.0
  sum_f1 = 0.0
  sum_precision_weighted = 0.0
  sum_recall_weighted = 0.0
  sum_f1_weighted = 0.0
  class_support = np.zeros((cnf_matrix.shape[0], 1))
  for i in range(cnf_matrix.shape[0]):
    tp = cnf_matrix[i, i]
    fp = np.sum(cnf_matrix[:, i]) - tp
    fn = np.sum(cnf_matrix[i, :]) - tp
    tn = np.sum(cnf_matrix) - tp - fp - fn
    class_support[i] = np.sum(cnf_matrix[i, :])

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * precision * recall / float(precision + recall)
    label = "class_" + str(i)
    class_report = {"label": label,
                    "precision": precision, "recall": recall, "f1": f1}
    sum_precision += precision
    sum_recall += recall
    sum_f1 += f1
    sum_precision_weighted += precision * class_support[i]
    sum_recall_weighted += recall * class_support[i]
    sum_f1_weighted += f1 * class_support[i]
    report += [class_report]

  # Compute Macro average (averaging the unweighted mean per label)
  class_num = float(cnf_matrix.shape[0])
  macro_avg_report = {"label": "macro_avg", "precision": sum_precision /
                      class_num, "recall": sum_recall / class_num, "f1": sum_f1 / class_num}
  report += [macro_avg_report]

  # Compute the weighted average (averaging the weighted mean per label)
  sum_weights = float(np.sum(class_support))
  weighted_avg_report = {"label": "weighted_avg",
                         "precision": float(sum_precision_weighted / sum_weights),
                         "recall": float(sum_recall_weighted / sum_weights),
                         "f1": float(sum_f1_weighted / sum_weights)}
  report += [weighted_avg_report]

  return report

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
  
  
def visualize_failure_predictions(failure_predictions: np.ndarray, masks: np.ndarray, ood_predictions: np.ndarray, rgb_images: np.ndarray, output_folder_path: str, session_nums: np.ndarray, image_names: list, unnormalize: bool = False, visualize_ood: bool = False, epistemic_unc_masks: np.ndarray = None, aleatoric_unc_masks: np.ndarray = None):
  """
  Colors the failure prediction for each pixel in the image and saves the resulting color image to file.
  :param failure_prediction: pixel-wise failure prediction (numpy array but in tensor shape form, i.e. (N, H, W))
  :param rgb_image: the input rgb image (N, C, H, W)
  :param masks: the input masks (N, 1, H, W)
  :param ood_predictions: the ood predictions (N, 1, H, W). The values should be binary
  :param session_nums: array of session numbers corresponding to each image (N, 1)
  :param epistemic_unc_masks: the ground-truth epistemic uncertainty masks (N, 1, H, W)
  :param aleatoric_unc_masks: the ground-truth aleatoric uncertainty masks (N, 1, H, W)
  :return:
  """
  alpha = 0.65
  normalize_mean = np.array([0.485, 0.456, 0.406])
  normalize_std = np.array([0.229, 0.224, 0.225])
  if unnormalize:
    normalize_mean_mat = np.zeros((rgb_images.shape[2],rgb_images.shape[3], rgb_images.shape[1]))
    normalize_std_mat = np.zeros((rgb_images.shape[2],rgb_images.shape[3], rgb_images.shape[1]))
    
    for j in range(normalize_mean_mat.shape[2]):
      normalize_mean_mat[:,:,j] = normalize_mean[j]
      normalize_std_mat[:,:,j] = normalize_std[j]
  
  masks = np.squeeze(masks, axis=1)
  if visualize_ood:
    ood_predictions = np.squeeze(ood_predictions, axis=1)
  failure_predictions = failure_predictions > 0.5
  
  visualize_epistemic_mask = True if epistemic_unc_masks is not None else False
  visualize_aleatoric_mask = True if aleatoric_unc_masks is not None else False
  
  assert failure_predictions.shape[0] == len(image_names)
  
  for i in range(failure_predictions.shape[0]):
    failure_prediction = failure_predictions[i, :, :]
    mask = masks[i, :, :]
    if visualize_epistemic_mask:
      mask_epistemic = epistemic_unc_masks[i, :, :]
    if visualize_aleatoric_mask:
      mask_aleatoric = aleatoric_unc_masks[i, :, :]
    if visualize_ood:
      ood_prediction = ood_predictions[i, :, :]
    rgb_image = rgb_images[i, :, :, :]
    image_idx = image_names[i][0:-4]
    session_idx = "{0:05d}".format(session_nums[i])

    tmp = np.logical_or(failure_prediction == 1, failure_prediction == 0)
    assert np.all(tmp), "Error: failure prediction is not in {0, 1}. failure_prediction = {}".format(
        failure_prediction)

    rgb_image = np.moveaxis(rgb_image, [0, 1, 2], [2, 0, 1])
    if unnormalize:
      rgb_image = rgb_image * normalize_std_mat + normalize_mean_mat
      rgb_image = (rgb_image * 255).astype(np.uint8)


    assert failure_prediction.shape[0] == rgb_image.shape[0], "Failure prediction and rgb image must have the same shape."
    assert failure_prediction.shape[1] == rgb_image.shape[1], "Failure prediction and rgb image must have the same shape."

    gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    gray_img_overlaid = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGRA)
    gray_img_overlaid_masked = gray_img_overlaid.copy()
    gray_img_overlaid_with_ood_masked = gray_img_overlaid.copy()

    if visualize_epistemic_mask:
      gray_img_overlaid_with_epistemic_masked = gray_img_overlaid.copy()
    if visualize_aleatoric_mask:
      gray_img_overlaid_with_aleatoric_masked = gray_img_overlaid.copy()

    failure_prediction_color = np.zeros(gray_img_overlaid.shape, dtype=np.uint8)
    failure_prediction_color[failure_prediction] = [0, 0, 255, 255]
    failure_prediction_color[np.logical_not(failure_prediction)] = [
        0, 255, 0, 255]
    
    if visualize_ood:
      failure_prediction_color_with_ood = failure_prediction_color.copy()
      failure_prediction_color_with_ood[ood_prediction] = [255, 0, 255, 255]
    
    # Generate the image with failure prediction overlaid
    cv2.addWeighted(failure_prediction_color, alpha, gray_img_overlaid, 1 - alpha, 0, gray_img_overlaid)

    # Generate the same image but color the masked area differently
    failure_prediction_color_masked = failure_prediction_color
    failure_prediction_color_masked[np.logical_not(mask)] = [255, 0, 0, 255]
    cv2.addWeighted(failure_prediction_color_masked, alpha, gray_img_overlaid_masked, 1 - alpha, 0, gray_img_overlaid_masked)
    
    # Generate the masked failure prediction visualization with ood predictions
    if visualize_ood:
      failure_prediction_color_ood_masked = failure_prediction_color_with_ood
      failure_prediction_color_ood_masked[np.logical_not(mask)] = [255, 0, 0, 255]
      cv2.addWeighted(failure_prediction_color_ood_masked, alpha, gray_img_overlaid_with_ood_masked, 1 - alpha, 0, gray_img_overlaid_with_ood_masked)
    
    if visualize_epistemic_mask:
      epistemic_mask_color = np.zeros(gray_img_overlaid.shape, dtype=np.uint8)
      epistemic_mask_color[mask_epistemic] = [255, 0, 255, 255]
      epistemic_mask_color[np.logical_not(mask_epistemic)] = [
          0, 255, 0, 255]
      epistemic_mask_color_masked = epistemic_mask_color
      epistemic_mask_color_masked[np.logical_not(mask)] = [255, 0, 0, 255]
      cv2.addWeighted(epistemic_mask_color_masked, alpha, gray_img_overlaid_with_epistemic_masked, 1 - alpha, 0, gray_img_overlaid_with_epistemic_masked)
      
    if visualize_aleatoric_mask:
      aleatoric_mask_color = np.zeros(gray_img_overlaid.shape, dtype=np.uint8)
      aleatoric_mask_color[mask_aleatoric] = [255, 0, 255, 255]
      aleatoric_mask_color[np.logical_not(mask_aleatoric)] = [
          0, 255, 0, 255]
      aleatoric_mask_color_masked = aleatoric_mask_color
      aleatoric_mask_color_masked[np.logical_not(mask)] = [255, 0, 0, 255]
      cv2.addWeighted(aleatoric_mask_color_masked, alpha, gray_img_overlaid_with_aleatoric_masked, 1 - alpha, 0, gray_img_overlaid_with_aleatoric_masked)
      
    
    # Create output folder if it does not exist
    output_path_vis = os.path.join(output_folder_path, str(session_idx), 'failure_pred_vis')
    output_path_vis_masked = os.path.join(output_folder_path, str(session_idx), 'failure_pred_masked_vis')
    output_path_vis_ood_masked = os.path.join(output_folder_path, str(session_idx), 'failure_pred_with_ood_masked_vis')
    output_path_vis_epistemic_masked = os.path.join(output_folder_path, str(session_idx), 'gt_epistemic_unc_vis')
    output_path_vis_aleatoric_masked = os.path.join(output_folder_path, str(session_idx), 'gt_aleatoric_unc_vis')
    if not os.path.exists(output_folder_path):
      os.makedirs(output_folder_path)
    if not os.path.exists(output_path_vis):
      os.makedirs(output_path_vis)
    if not os.path.exists(output_path_vis_masked):
      os.makedirs(output_path_vis_masked)
    if not os.path.exists(output_path_vis_ood_masked) and visualize_ood:
      os.makedirs(output_path_vis_ood_masked)
    if not os.path.exists(output_path_vis_epistemic_masked) and visualize_epistemic_mask:
      os.makedirs(output_path_vis_epistemic_masked)
    if not os.path.exists(output_path_vis_aleatoric_masked) and visualize_aleatoric_mask:
      os.makedirs(output_path_vis_aleatoric_masked)
    output_image_path = os.path.join(output_path_vis, image_idx + ".png")
    output_image_masked_path = os.path.join(output_path_vis_masked, image_idx + ".png")
    output_image_ood_masked_path = os.path.join(output_path_vis_ood_masked, image_idx + ".png")
    output_image_epistemic_masked_path = os.path.join(output_path_vis_epistemic_masked, image_idx + ".png")
    output_image_aleatoric_masked_path = os.path.join(output_path_vis_aleatoric_masked, image_idx + ".png")
    cv2.imwrite(output_image_path,
                gray_img_overlaid)
    cv2.imwrite(output_image_masked_path,
                gray_img_overlaid_masked)
    if visualize_ood:
      cv2.imwrite(output_image_ood_masked_path,
                gray_img_overlaid_with_ood_masked)
    if visualize_epistemic_mask:
      cv2.imwrite(output_image_epistemic_masked_path,
                gray_img_overlaid_with_epistemic_masked)
    if visualize_aleatoric_mask:
      cv2.imwrite(output_image_aleatoric_masked_path,
                gray_img_overlaid_with_aleatoric_masked)
  
def convert_image_name_to_index(image_name):
  """
  Converts the image name to an index.
  :param image_name
  :return: index
  """
  image_index = image_name.rstrip(".png")

  return image_index
 
def visualize_scalar_img_on_rgb(scalar_img: np.ndarray, rgb_image: np.ndarray, output_folder_path: str, image_idx: str, max_unc_threshold: float =10.0, unnormalize: bool =False):
  """
  Visualizes the magnitude of the input scalar image for each pixel in the image.
  :param rgb_image: the input rgb image np.array (1, C, H, W)
  :param scalar_img: the input masks np.array (1, 1, H, W)
  :return:
  """
  
  normalize_mean = np.array([0.485, 0.456, 0.406])
  normalize_std = np.array([0.229, 0.224, 0.225])
  if unnormalize:
    normalize_mean_mat = np.zeros((rgb_image.shape[2],rgb_image.shape[3], rgb_image.shape[1]))
    normalize_std_mat = np.zeros((rgb_image.shape[2],rgb_image.shape[3], rgb_image.shape[1]))
    
    for j in range(normalize_mean_mat.shape[2]):
      normalize_mean_mat[:,:,j] = normalize_mean[j]
      normalize_std_mat[:,:,j] = normalize_std[j]
  
  alpha = 0.5
  MAX_UNC_THRESH_VISUALIZATION = max_unc_threshold

  # Remove the first axis (the batch dimension for the rgb image and the batch and channel dimensions for the scalar image)
  scalar_img = np.squeeze(scalar_img, axis=(0,1))
  rgb_image = np.squeeze(rgb_image, axis=0)
  
  rgb_image = np.moveaxis(rgb_image, [0, 1, 2], [2, 0, 1])
  if unnormalize:
    rgb_image = rgb_image * normalize_std_mat + normalize_mean_mat
    rgb_image = (rgb_image * 255).astype(np.uint8)

  rgb_image = (cv2.resize(
      rgb_image,
      (scalar_img.shape[1], scalar_img.shape[0]))
      * 1).astype(np.uint8)

  assert scalar_img.shape[0] == rgb_image.shape[0], "Scalar img and rgb image must have the same shape."
  assert scalar_img.shape[1] == rgb_image.shape[1], "Scalar img and rgb image must have the same shape."

  gray_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
  gray_img_overlaid = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGBA)

  unc_img = colorize(
      scalar_img, plt.get_cmap('viridis'), 0, MAX_UNC_THRESH_VISUALIZATION)
  unc_img = np.uint8(unc_img * 255)
  unc_img = cv2.cvtColor(unc_img, cv2.COLOR_RGBA2BGRA)

  gray_img_overlaid = gray_img_overlaid.astype(
      np.uint8)
  cv2.addWeighted(unc_img,
                  alpha, gray_img_overlaid, 1 - alpha,
                  0, gray_img_overlaid)

  # Create output folder if it does not exist
  if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
  output_image_path = os.path.join(output_folder_path, image_idx + ".png")
  cv2.imwrite(output_image_path,
              gray_img_overlaid)
  
  
def resize_images(images: torch.Tensor, desired_size: tuple) -> torch.Tensor:
  """
  Resizes the images to the desired size.
  :param images: tensor of shape (N, C, H, W)
  :param desired_size: tuple of (W, H)
  :return:
  """
  images_np = images.cpu().numpy()
  
  images_resized = torch.zeros(images.shape[0], images.shape[1], desired_size[1], desired_size[0], dtype=images.dtype)
  images_resized_np = images_resized.numpy()
  
  images_np = np.transpose(images_np, (0, 2, 3, 1))
  images_resized_np = np.transpose(images_resized_np, (0, 2, 3, 1))
  
  for i in range(images_resized_np.shape[0]):
    images_resized_np[i,:,:,:] = cv2.resize(images_np[i,:,:,:], desired_size)
  
  return torch.from_numpy(np.transpose(images_resized_np, (0, 3, 1, 2))).to(images.device)


# Helper function for saving inference results
def save_result_images(input_imgs,
                       target_imgs,
                       output_imgs,
                       img_names,
                       session_nums,
                       save_dir,
                       raw_output_size,
                       gt_available = True,
                       save_raw_output = True,
                       initial_directory_prep = False,
                       gt_masks = None,
                       pred_masks = None):
  save_masked_imgs = (gt_masks is not None) and gt_available
  img_num = input_imgs.shape[0]

  for i in range(img_num):
    # Save the output image along with ground truth and input image
    fig = plt.figure()
    # grid = AxesGrid(fig, 111,
    #                 nrows_ncols=(1, 3),
    #                 axes_pad=0.05,
    #                 cbar_mode='single',
    #                 cbar_location='right',
    #                 cbar_pad=0.1)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.05)

    for ax in grid:
      ax.set_axis_off()
    ax = grid[0]
    ax.imshow(input_imgs[i, 0, :, :], cmap='gray')

    if gt_available:
      ax = grid[1]
      ax.imshow(target_imgs[i, 0, :, :], vmin=0.0, vmax=1.0, cmap='viridis')
      # ax.imshow(target_imgs[i, 0,:,:], cmap='viridis')
    ax = grid[2]
    im = ax.imshow(output_imgs[i, -1, :, :], vmin=0.0, vmax=1.0, cmap='viridis')
    # im = ax.imshow(output_imgs[i, 0,:,:], cmap='viridis')

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]
    # cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)

    session_folder = "{0:05d}".format(session_nums[i])

    # Rescale the raw output of the network and save it as a grayscale jpg image
    if save_raw_output:
      raw_output_dir = save_dir + '/' + session_folder
      raw_output_path = raw_output_dir + '/' + img_names[i][:-4] + '.jpg'
      if not os.path.exists(raw_output_dir):
        os.makedirs(raw_output_dir)
      elif initial_directory_prep:
        shutil.rmtree(raw_output_dir)
        os.makedirs(raw_output_dir)

      resized_img = cv2.resize(output_imgs[i, -1, :, :], raw_output_size)
      # scipy.misc.toimage(resized_img, cmin=0.0, cmax=1.0).save(raw_output_path)
      plt.imsave(raw_output_path, resized_img, cmap='gray', vmin=0.0, vmax=1.0)

    # If ground truth is available visualize input, gt, and network output
    # next to each other and save them to file. If gt not available only
    # visualize input and network output. The target directory is different
    # for these two cases
    if gt_available:
      vis_dir = save_dir + '/vis_with_gt/' + session_folder
      vis_path = vis_dir + '/' + img_names[i]

      if save_masked_imgs:
        vis_dir_masked = save_dir + '/vis_with_gt_mask/' + session_folder
        vis_path_masked = vis_dir_masked + '/' + img_names[i]
    else:
      vis_dir = save_dir + '/vis_wo_gt/' + session_folder
      vis_path = vis_dir + '/' + img_names[i]

    if not os.path.exists(vis_dir):
      os.makedirs(vis_dir)
    elif initial_directory_prep:
      shutil.rmtree(vis_dir)
      os.makedirs(vis_dir)

    if save_masked_imgs:
      if not os.path.exists(vis_dir_masked):
        os.makedirs(vis_dir_masked)
      elif initial_directory_prep:
        shutil.rmtree(vis_dir_masked)
        os.makedirs(vis_dir_masked)

    plt.savefig(vis_path, dpi=150)

    if save_masked_imgs:
      ax = grid[1]
      target_img = cm.viridis(target_imgs[i, 0, :, :].astype(np.float32))
      mask_bin = gt_masks[i, 0, :, :] > 0.5
      mask_arg = np.nonzero(np.logical_not(mask_bin))
      for k in range(3):
        curr_color = target_img[:,:,k]
        curr_color[mask_arg] = 0
      ax.imshow(target_img)

      ax = grid[2]
      output_img = cm.viridis(output_imgs[i, -1, :, :])
      if pred_masks is not None:
        mask_bin = pred_masks[i, 0, :, :] > 0.5
        mask_arg = np.nonzero(np.logical_not(mask_bin))
        for k in range(3):
          curr_color = output_img[:,:,k]
          curr_color[mask_arg] = 0
      ax.imshow(output_img)

      plt.savefig(vis_path_masked, dpi=150)

    plt.close()



def calculate_gpu_usage(gpus):
  total_usage_mb = 0.0
  for i in range(len(gpus)):
    total_usage_mb += float(torch.cuda.max_memory_allocated(gpus[i]) +
                        torch.cuda.max_memory_cached(gpus[i])) / 1000000.0
  return total_usage_mb


def main(cfg, args, gpus):
  DETECT_OOD_BASED_ON_ENTROPY = True # if set to False then OOD is detected based on prediction variance.
  ENTROPY_THRESH = 0.1
  # PREDICTION_VARIANCE_THRESH = 0.0005
  PREDICTION_VARIANCE_THRESH = 100.0
  CONFUSION_MATRIX_COMPUTATION_BATCH_SIZE = 100
  SAVE_VISUALIZATIONS = True
  SAVE_ENTROPY_VALUES = True
  USE_MULTI_GPU = True if len(gpus) > 1 else False
  NUM_WORKERS = cfg.TEST.workers
  if cfg.TEST.use_gpu:
    BATCH_SIZE = cfg.TEST.batch_size_per_gpu * len(gpus)
  else:
    BATCH_SIZE = cfg.TEST.batch_size

  # TODO: Support batch-size of larger than 1.
  if BATCH_SIZE > 1:
    print("ERROR - Batches of size > 1 are not supported.")
    exit()

  RESULT_SAVE_DIR = cfg.TEST.result + '/' + cfg.MODEL.name + '/'
  if not os.path.exists(RESULT_SAVE_DIR):
    os.makedirs(RESULT_SAVE_DIR)
  if not os.path.exists(RESULT_SAVE_DIR):
    print("Error: Could not create result directory ", RESULT_SAVE_DIR)
    exit()

  test_set_dict = {
    "test_tmp":[1007],
    "test_01_ganet_v0": [1007, 1012, 1017, 1022, 1027, 1032, 2007, 2012, 2017, 2022, 2027, 2032],
    "test_ood_01_ganet_v0": [3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036],
    "test_ood_N_01_ganet_v0": [1000, 1001, 1002, 1003],
    "test_ood_africa_01_ganet_v0": [1000, 1001, 1002, 1003, 1004],
    "test_ood_africa_02_ganet_v0": [1010, 1011, 1012, 1013]
  }

  class_weights_dict = {
    "test_01_ganet_v0": torch.tensor(
      [5834415.0 / 233773276.0, 227938861.0 / 233773276.0], dtype=torch.float32),  # NF, F
    "test_ood_N_01_ganet_v0": torch.tensor(
      [8731419.0 / 184377799.0, 175646380.0 / 184377799.0], dtype=torch.float32),  # NF, F
    "test_ood_africa_01_ganet_v0": torch.tensor(
      [48456861.0 / 134077512.0, 85620651.0 / 134077512.0], dtype=torch.float32),  # NF, F
    "test_ood_africa_02_ganet_v0": torch.tensor(
      [7418926.0 / 115252778.0, 107833852.0 / 115252778.0], dtype=torch.float32),  # NF, F 
  }

  session_list_test = [7, 9, 10]


  if cfg.DATASET.test_set is not None:
    session_list_test = test_set_dict[cfg.DATASET.test_set]

  if cfg.TEST.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:"+str(gpus[0]))
    used_gpu_count = 1
    total_mem =(
          float(torch.cuda.get_device_properties(device).total_memory)
          / 1000000.0)
    gpu_name = torch.cuda.get_device_name(device)
    print("Using ", gpu_name, " with ", total_mem, " MB of memory.")
  else:
    device = torch.device("cpu")
    used_gpu_count = 0


  print("Output Model: ", cfg.MODEL.name)
  print("Test data: ", session_list_test)
  print("Network base model is ", cfg.MODEL.arch_encoder  + '+' +
         cfg.MODEL.arch_decoder)
  print("Batch size: ", BATCH_SIZE)
  print("Workers num: ", NUM_WORKERS)
  print("device: ", device)
  phases = ['test']
  #data_set_portion_to_sample = {'train': 0.8, 'val': 0.2}
  data_set_portion_to_sample = {'train': 1.0, 'val': 1.0}
  
  assert cfg.DATASET.test_set in class_weights_dict, "Class weights not found for test set {}".format(
    cfg.DATASET.test_set)  
  class_weights = class_weights_dict[cfg.DATASET.test_set]
  


  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  input_img_width = int(cfg.DATASET.img_width)
  input_img_height = int(cfg.DATASET.img_height)
  target_img_width = int(cfg.TEST.output_img_width)
  target_img_height = int(cfg.TEST.output_img_height)

  # Transform loaded images. If not using color images, it will copy the single
  # channel 3 times to keep the size of an RGB image.
  if cfg.DATASET.use_color_images:
    if cfg.DATASET.normalize_input :
      data_transform_input = transforms.Compose([
                transforms.Resize((input_img_height, input_img_width)),
                transforms.ToTensor(),
                normalize
            ])
    else:
      data_transform_input = transforms.Compose([
              transforms.Resize((input_img_height, input_img_width)),
              transforms.ToTensor(),
          ])
  else:
    if cfg.DATASET.normalize_input :
      data_transform_input = transforms.Compose([
                transforms.Resize((input_img_height, input_img_width)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                normalize
            ])
    else:
      data_transform_input = transforms.Compose([
              transforms.Resize((input_img_height, input_img_width)),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
          ])

  data_transform_target = transforms.Compose([
            transforms.Resize((target_img_height, target_img_width)),
            transforms.ToTensor()
        ])
  load_mask = cfg.TRAIN.use_masked_loss or cfg.MODEL.predict_conf_mask
  
  root_refined_model_dir = cfg.DATASET.root_refined_model if cfg.DATASET.root_refined_model!="" else None
  
  loading_refined_model_labels = True if (root_refined_model_dir is not None) and cfg.TEST.ground_truth_available else False
  
  test_dataset = DepthErrorDataset(cfg.DATASET.root,
                                  cfg.DATASET.raw_img_root,
                                  session_list_test,
                                  root_refined_model_dir=root_refined_model_dir,
                                  loaded_image_color=cfg.DATASET.is_dataset_color,
                                  output_image_color=cfg.DATASET.use_color_images,
                                  session_prefix_length=cfg.DATASET.session_prefix_len,
                                  raw_img_folder=cfg.DATASET.raw_img_folder,
                                  raw_img_folder_second_camera=cfg.DATASET.raw_img_folder_second_camera,
                                  label_img_folder=cfg.DATASET.label_img_folder,
                                  mask_img_folder=cfg.DATASET.mask_img_folder,
                                  transform_input=data_transform_input,
                                  transform_target=data_transform_target,
                                  load_masks=load_mask,
                                  regression_mode=cfg.MODEL.is_regression_mode,
                                  binarize_target=cfg.DATASET.binarize_target,
                                  no_meta_data_available=True,
                                  load_only_with_labels=cfg.TEST.ground_truth_available,
                                  stereo_mode=cfg.MODEL.is_stereo)
  datasets = {phases[0]: test_dataset}

  data_loaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS)
                  for x in phases}

  # The desired size of the output image. The model interpolates the output
  # to this size
  desired_size = (cfg.DATASET.img_height,
                  cfg.DATASET.img_width)

  # The desired size for the output of the network when saving it to file
  # as an image
  raw_output_img_size = (cfg.TEST.output_img_width,
                         cfg.TEST.output_img_height)


  if cfg.MODEL.is_regression_mode:
      if cfg.TRAIN.use_masked_loss:
        criterion = MaskedMSELoss()
        print("Regression Mode with Masked Loss")
      else:
        criterion = nn.MSELoss(reduction='mean')
        print("Regression Mode")
  else:
    criterion = nn.NLLLoss(weight=class_weights, ignore_index=-1)
    print("Classification Mode")

  # Load networks
  nets = []
  for encoder, decoder in zip(cfg.MODEL.weights_encoder, 
                              cfg.MODEL.weights_decoder):

    # Build the network from selected modules
    net_encoder = ModelBuilder.build_encoder(
      arch=cfg.MODEL.arch_encoder.lower(),
      fc_dim=cfg.MODEL.fc_dim,
      weights=encoder)
    net_decoder = ModelBuilder.build_decoder(
      arch=cfg.MODEL.arch_decoder.lower(),
      fc_dim=cfg.MODEL.fc_dim,
      num_class=cfg.DATASET.num_class,
      weights=decoder,
      regression_mode=cfg.MODEL.is_regression_mode,
      inference_mode=True)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
      net = SegmentationModule(
        net_encoder, net_decoder, criterion, cfg.TRAIN.deep_sup_scale,
        segSize=desired_size)
    else:
      net = SegmentationModule(
        net_encoder, net_decoder, criterion, segSize=desired_size)
    
    nets += [net]
    

  # TODO: TEMPORARY Debugging
  param_size = 0
  for param in nets[0].parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in nets[0].buffers():
      buffer_size += buffer.nelement() * buffer.element_size()
  size_all_mb = (param_size + param_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))


  if cfg.TEST.use_gpu and USE_MULTI_GPU:
    if torch.cuda.device_count() >= len(gpus):
      available_gpu_count = torch.cuda.device_count()
      print("Using ", len(gpus), " GPUs out of available ", available_gpu_count)
      print("Used GPUs: ", gpus)
      for net in nets:
        net = nn.DataParallel(net, device_ids=gpus)
      # For synchronized batch normalization:
      patch_replication_callback(net)
    else:
      print("Requested GPUs not available: ", gpus)
      exit()

  for net in nets:
    net = net.to(device)
    net.eval()

  print("Starting Inference...")
  start_time = time.time()

  timings_individual_all = []
  warm_up_iterations = 30
  if cfg.TEST.measure_inference_time:
    starter, ender = torch.cuda.Event(
    enable_timing=True), torch.cuda.Event(enable_timing=True)

  all_predictions = np.array([], dtype=np.int_)
  all_binary_labels = np.array([], dtype=np.int_)
  cnf_matrix = np.zeros((2, 2), dtype=np.int_)
  # predictions, labels, and confusion matrix for subset of the data that
  # does not include any instances of epistemic uncertainty
  all_predictions_non_epistemic = np.array([], dtype=np.int_)
  all_binary_labels_non_epistemic = np.array([], dtype=np.int_)
  cnf_matrix_non_epistemic = np.zeros((2, 2), dtype=np.int_)
  # predictions, labels, and confusion matrix for subset of the data that
  # does not include any instances of aleatoric uncertainty
  all_predictions_non_aleatoric = np.array([], dtype=np.int_)
  all_binary_labels_non_aleatoric = np.array([], dtype=np.int_)
  cnf_matrix_non_aleatoric = np.zeros((2, 2), dtype=np.int_)
  valid_data_points_count = 0

  pred_ood_in_range_count = 0 # Counts the number of predicted OOD data points that were also within the valid depth range
  pred_failure_in_range_count = 0 # Counts the number of predicted failure data points that were also within the valid depth range
  
  # Stores the entropy for all predictions that correspond to in-range depth values
  all_masked_entropy = np.array([], dtype=np.float_)
  
  session_name_format = "{0:05d}"

  # Runs inference on all data
  for cur_phase in phases:
    # Set model to evaluate mode
   
    running_loss = 0.0
    # Iterate over data
    for i, data in enumerate(tqdm(data_loaders[cur_phase]), 0):  
      # get the inputs
      input = data['img']
      img_names = data['img_name']
      session_nums = data['session']
      feed_dict = dict()
      feed_dict['input'] = input.to(device)

      img_ids = [convert_image_name_to_index(img_name) for img_name in img_names]
      

      mask_np = None
      if cfg.TRAIN.use_masked_loss:
        mask = data['mask_img']
        feed_dict['mask'] = mask.to(device)
        mask_np = mask.numpy()
      if not cfg.MODEL.is_regression_mode:
        target = data['labels']
        feed_dict['target'] = target.to(device)
        
        if loading_refined_model_labels:
          target_refined_model = data['labels_refined_model']
          gt_epistemic_unc_mask = torch.logical_and(target_refined_model == 0, target == 1)
          # gt_aleatoric_unc_mask = target_refined_model == 1
          gt_aleatoric_unc_mask = torch.logical_and(target_refined_model == 1, target == 1)
          
          gt_non_epistemic_unc_mask = torch.logical_not(gt_epistemic_unc_mask)
          gt_non_aleatoric_unc_mask = torch.logical_not(gt_aleatoric_unc_mask)
      else:
        print("ERROR: Regression mode not supported")
        exit()

      # Do not track history since we are in eval mode
      with torch.set_grad_enabled(False):
        if cfg.TEST.is_ensemble:
          ensemble_pred_size = tuple([len(nets)] + [target.size()[0]] + [cfg.DATASET.num_class] + list(target.size()[1:]))
          ensemble_prediction = torch.zeros(ensemble_pred_size, dtype=torch.float32, device=feed_dict['target'].device)
          
          j = 0
          for net in nets:
            output = net(feed_dict)
            
            # Resize the output to the size of original RGB images
            input = resize_images(input, raw_output_img_size)  
            output = resize_images(output, raw_output_img_size)

            ensemble_prediction[j, :, :, :] = output
            j += 1
          prediction = torch.mean(ensemble_prediction, axis=0)
          prediction_variance = torch.mean(torch.var(ensemble_prediction, axis=0), axis=1)
          
                    
          if cfg.TRAIN.use_masked_loss and cfg.MODEL.is_regression_mode:
            loss = criterion(prediction, feed_dict['target'], feed_dict['mask'])
          else:  
            loss = criterion(prediction, feed_dict['target'])
          if not cfg.MODEL.is_regression_mode:
            target = torch.reshape(target, (target.size()[0], 
                                            1,
                                            target.size()[1],
                                            target.size()[2]))
            
          # Compute Entropy
          # TODO: Handle batch_size > 1
          entropy = torch.sum(-prediction * torch.log(prediction + 1e-15), dim=1)
          entropy_mask = entropy > ENTROPY_THRESH # 1 X H X W
          entropy.unsqueeze_(1)
          
          prediction_var_mask = prediction_variance > PREDICTION_VARIANCE_THRESH # 1 X H X W
          prediction_variance.unsqueeze_(1)

          masked_entropy = entropy[feed_dict['mask']]
          all_masked_entropy = np.concatenate((all_masked_entropy, masked_entropy.cpu().numpy()))
          
          # Switch between entropy_mask and prediction_var_mask
          if DETECT_OOD_BASED_ON_ENTROPY:
            ood_mask = entropy_mask
          else:
            ood_mask = prediction_var_mask
            
          prediction_label = torch.argmax(prediction, dim=1)
          
          ood_mask.unsqueeze_(1)
          pred_ood_in_range_count += torch.sum(ood_mask[feed_dict['mask']])
          
        else:
          if cfg.TEST.measure_inference_time and i > warm_up_iterations:
            starter.record()
          
          # forward pass
          output = net(feed_dict)
          
          if cfg.TEST.measure_inference_time and i > warm_up_iterations:
            ender.record()
            torch.cuda.synchronize()
            timings_individual_all += [starter.elapsed_time(ender)]
          
          # Resize the output to the size of original RGB images
          input = resize_images(input, raw_output_img_size)  
          output = resize_images(output, raw_output_img_size)
        
          if cfg.TRAIN.use_masked_loss and cfg.MODEL.is_regression_mode:
            loss = criterion(output, feed_dict['target'], feed_dict['mask'])
          else:  
            loss = criterion(output, feed_dict['target'])

          if not cfg.MODEL.is_regression_mode:
            target = torch.reshape(target, (target.size()[0], 
                                            1,
                                            target.size()[1],
                                            target.size()[2]))
          
          prediction_label = torch.argmax(output, dim=1)
          
          # Empty dummy tensor (this is only used in the ensemble mode)
          ood_mask = torch.tensor([], dtype=torch.float32, device=feed_dict['target'].device)

        
        if cfg.TRAIN.use_masked_loss:
          curr_labels = target[feed_dict['mask']].cpu().numpy().astype(np.int_)
          current_pred = prediction_label[torch.squeeze(feed_dict['mask'], 1)].cpu().numpy().astype(np.int_)
          pred_failure_in_range_count += np.sum(current_pred)
          all_predictions = np.concatenate((all_predictions, current_pred), 0) 
          all_binary_labels = np.concatenate((all_binary_labels, curr_labels), 0)
          
          # Generate different versions of the labels and predictions via masking out aleatoric and epistemic uncertainty           
          if loading_refined_model_labels:
            curr_mask = torch.logical_and(gt_non_epistemic_unc_mask, mask)
            curr_labels_non_epistemic = target[curr_mask].numpy().astype(np.int_)
            current_pred_non_epistemic = prediction_label[torch.squeeze(curr_mask, 1)].cpu().numpy().astype(np.int_)
            all_predictions_non_epistemic = np.concatenate((all_predictions_non_epistemic, current_pred_non_epistemic), 0) 
            all_binary_labels_non_epistemic = np.concatenate((all_binary_labels_non_epistemic, curr_labels_non_epistemic), 0)
            
            curr_mask = torch.logical_and(gt_non_aleatoric_unc_mask, mask)
            curr_labels_non_aleatoric = target[curr_mask].numpy().astype(np.int_)
            current_pred_non_aleatoric = prediction_label[torch.squeeze(curr_mask, 1)].cpu().numpy().astype(np.int_)
            all_predictions_non_aleatoric = np.concatenate((all_predictions_non_aleatoric, current_pred_non_aleatoric), 0) 
            all_binary_labels_non_aleatoric = np.concatenate((all_binary_labels_non_aleatoric, curr_labels_non_aleatoric), 0)
          
          
        if i % CONFUSION_MATRIX_COMPUTATION_BATCH_SIZE == 0 and i > 0:
          # Compute the confusion matrix for current batch and add it to the total confusion matrix
          cnf_matrix = cnf_matrix + confusion_matrix(
              all_binary_labels, all_predictions)
          print("Current confusion matrix:")
          print(cnf_matrix)

          # Reset the arrays
          valid_data_points_count += all_predictions.size
          all_predictions = np.array([], dtype=np.int_)
          all_binary_labels = np.array([], dtype=np.int_)
          
          if loading_refined_model_labels:
            cnf_matrix_non_epistemic = cnf_matrix_non_epistemic + confusion_matrix(
                all_binary_labels_non_epistemic, all_predictions_non_epistemic)
            print("Current non epistemic confusion matrix:")
            print(cnf_matrix_non_epistemic)
            
            cnf_matrix_non_aleatoric = cnf_matrix_non_aleatoric + confusion_matrix(
                all_binary_labels_non_aleatoric, all_predictions_non_aleatoric)
            print("Current non aleatoric confusion matrix:")
            print(cnf_matrix_non_aleatoric)

            # Reset the arrays
            all_predictions_non_epistemic = np.array([], dtype=np.int_)
            all_binary_labels_non_epistemic = np.array([], dtype=np.int_)
            all_predictions_non_aleatoric = np.array([], dtype=np.int_)
            all_binary_labels_non_aleatoric = np.array([], dtype=np.int_)
            
          
          # Save the entropy data to file
          if SAVE_ENTROPY_VALUES:
          entropy_out_dir = os.path.join(RESULT_SAVE_DIR, 'entropy')
          if not os.path.exists(entropy_out_dir):
            os.makedirs(entropy_out_dir)
          entropy_file_path = os.path.join(entropy_out_dir, str(i/CONFUSION_MATRIX_COMPUTATION_BATCH_SIZE) + '.npy')
          np.save(entropy_file_path , all_masked_entropy)
          
          print("Saved entropy data to file: {}".format(entropy_file_path))
          
          # Reset the entropy array
          all_masked_entropy = np.array([], dtype=np.float_)
          

        
      ##TODO: TEMPORARY Debugging
      # mem_c = torch.cuda.memory_allocated()
      # print("Current Memory allocated (MB): ", mem_c * 1e-6)


      input_np = input.to(torch.device("cpu")).numpy()
      target_np = target.to(torch.device("cpu")).numpy()
      output_np = output.to(torch.device("cpu")).numpy()
      
      # Only use the first three channels of the input image for visualizations since in the case of stereo inputs, channels 4-6 correspond to the secondary image
      input_np = input_np[:, :3, :, :]
      
      if SAVE_VISUALIZATIONS:
        if not loading_refined_model_labels:
        visualize_failure_predictions(prediction_label.cpu().numpy(), mask_np, ood_mask.cpu().numpy(), input_np, RESULT_SAVE_DIR, session_nums.cpu().numpy(), img_names, unnormalize=True, visualize_ood=cfg.TEST.is_ensemble)
        else:
          visualize_failure_predictions(prediction_label.cpu().numpy(), mask_np, ood_mask.cpu().numpy(), input_np, RESULT_SAVE_DIR, session_nums.cpu().numpy(), img_names, unnormalize=True, visualize_ood=cfg.TEST.is_ensemble, epistemic_unc_masks=gt_epistemic_unc_mask.numpy(), aleatoric_unc_masks=gt_aleatoric_unc_mask.numpy())
        
        
      
        if cfg.TEST.is_ensemble:
      # TODO: Handle batch_size > 1. Currently only visuzlize the first image in the batch
      # TODO: tune max unc threshold param
      session_name = "{0:05d}".format(session_nums[0])
      visualize_scalar_img_on_rgb(
        entropy.cpu().numpy(), input_np, os.path.join(RESULT_SAVE_DIR, session_name, 'prediction_entropy_vis'), img_ids[0], max_unc_threshold=ENTROPY_THRESH, unnormalize=True)
        
        visualize_scalar_img_on_rgb(
        prediction_variance.cpu().numpy(), input_np, os.path.join(RESULT_SAVE_DIR, session_name, 'prediction_variance_vis'), img_ids[0], max_unc_threshold=PREDICTION_VARIANCE_THRESH, unnormalize=True)
      

      # # TODO: TEMPORARY Debugging
      # print("Loss size: ", loss.size())
      # print("Loss: ", loss)
      loss = loss.item()
      running_loss += loss


      if cur_phase == 'test':
        if i % 100 == 99:    # print every 100 mini-batches
          print('[%5d] Loss: %.6f' %
              (i + 1, running_loss / i))

          if used_gpu_count:
            print("Total GPU usage (MB): ",
              calculate_gpu_usage(gpus), " / ",
              used_gpu_count * total_mem)

  if cfg.TEST.measure_inference_time:
        mean_inf_time = np.mean(np.array(timings_individual_all))
        std_inf_time = np.std(np.array(timings_individual_all))
        print("Mean inference time for individual models: {:.4f} ms, Std: {:.4f} ms".format(
            mean_inf_time, std_inf_time))

  print('All data was processed.')
  time_elapsed = time.time() - start_time
  print('Completed in {:.0f}h {:.0f}m {:.0f}s'.format(
      time_elapsed//3600, (time_elapsed % 3600)//60, time_elapsed % 60))

  print('Total loss: ', running_loss / i)
  print('Total iteration', i)
  
  # Evalute the failure predictions
  if all_binary_labels.size > 0:
    cnf_matrix = cnf_matrix + confusion_matrix(all_binary_labels,
                                               all_predictions)
    valid_data_points_count += all_predictions.size
    
    if loading_refined_model_labels:
      cnf_matrix_non_epistemic = cnf_matrix_non_epistemic + confusion_matrix(
          all_binary_labels_non_epistemic, all_predictions_non_epistemic)
      
      cnf_matrix_non_aleatoric = cnf_matrix_non_aleatoric + confusion_matrix(
          all_binary_labels_non_aleatoric, all_predictions_non_aleatoric)
    
  report = compute_classification_report_from_cnf(cnf_matrix)
  print("Classification Report: ")
  print(report)
  # Save classification report to file
  report_file_path = os.path.join(cfg.TEST.result, 'classification_report.csv')
  with open(report_file_path, 'w') as csvfile:
    print("Writing classification report to file: " + report_file_path)
    writer = csv.DictWriter(csvfile, fieldnames=report[0].keys())
    writer.writeheader()
    writer.writerows(report)
    
  if loading_refined_model_labels:
    report_non_epistemic = compute_classification_report_from_cnf(cnf_matrix_non_epistemic)
    report_non_aleatoric = compute_classification_report_from_cnf(cnf_matrix_non_aleatoric)
    print("Classification Report Excluding Epistemic Uncertainty: ")
    print(report_non_epistemic)
    print("Classification Report Excluding Aleatoric Uncertainty: ")
    print(report_non_aleatoric)
    # Save classification report to file
    report_non_epistemic_file_path = os.path.join(cfg.TEST.result, 'classification_report_non_epistemic.csv')
    report_non_aleatoric_file_path = os.path.join(cfg.TEST.result, 'classification_report_non_aleatoric.csv')
    with open(report_non_epistemic_file_path, 'w') as csvfile:
      print("Writing classification report (excluding epistemic unc.) to file: " + report_non_epistemic_file_path)
      writer = csv.DictWriter(csvfile, fieldnames=report[0].keys())
      writer.writeheader()
      writer.writerows(report_non_epistemic)
    with open(report_non_aleatoric_file_path, 'w') as csvfile:
      print("Writing classification report (excluding aleatoric unc.) to file: " + report_non_aleatoric_file_path)
      writer = csv.DictWriter(csvfile, fieldnames=report[0].keys())
      writer.writeheader()
      writer.writerows(report_non_aleatoric)
    
    
  # Save the entropy data to file
  if SAVE_ENTROPY_VALUES:
  entropy_out_dir = os.path.join(RESULT_SAVE_DIR, 'entropy')
  if not os.path.exists(entropy_out_dir):
    os.makedirs(entropy_out_dir)
  entropy_file_path = os.path.join(entropy_out_dir, '0.npy')
  np.save(entropy_file_path , all_masked_entropy)
  print("Saved entropy data to file: {}".format(entropy_file_path))
    
  # Save confusion matrix to file
  cnf_file_path = os.path.join(cfg.TEST.result, 'confusion_mat_binary.csv')
  print("Writing confusion matrix to file: " + cnf_file_path)
  np.savetxt(cnf_file_path, cnf_matrix, delimiter=",", fmt=['%d', '%d'])
  
  if loading_refined_model_labels:
    cnf_file_path_non_epistemic = os.path.join(cfg.TEST.result, 'confusion_mat_non_epistemic_binary.csv')
    print("Writing confusion matrix to file: " + cnf_file_path_non_epistemic)
    np.savetxt(cnf_file_path_non_epistemic, cnf_matrix_non_epistemic, delimiter=",", fmt=['%d', '%d'])
    
    cnf_file_path_non_aleatoric = os.path.join(cfg.TEST.result, 'confusion_mat_non_aleatoric_binary.csv')
    print("Writing confusion matrix to file: " + cnf_file_path_non_aleatoric)
    np.savetxt(cnf_file_path_non_aleatoric, cnf_matrix_non_aleatoric, delimiter=",", fmt=['%d', '%d'])
      

  # Save the total loss to file
  loss_file_path = os.path.join(cfg.TEST.result, 'NLL_loss.txt')
  with open(loss_file_path, 'w') as f:
    f.write('NLL loss: {}\n'.format(running_loss / i))    
    f.write('Total valid (within depth range) data points: {}\n'.format(valid_data_points_count))
    f.write("Total predicted OOD data points: {}\n".format(pred_ood_in_range_count.item()))
    f.write("Total predicted failure data points: {}\n".format(pred_failure_in_range_count))

  print("Total valid data points: ", valid_data_points_count)
  print("Confusion matrix:")
  print(cnf_matrix)
  
  plot_confusion_matrix(
            cnf_matrix,
            ['NF', 'F'],
            "confusion_mat_binary", cfg.TEST.result,
            normalize=True)
  if loading_refined_model_labels:
    plot_confusion_matrix(
              cnf_matrix_non_epistemic,
              ['NF', 'F'],
              "confusion_mat_non_epistemic_binary", cfg.TEST.result,
              normalize=True)
    plot_confusion_matrix(
              cnf_matrix_non_aleatoric,
              ['NF', 'F'],
              "confusion_mat_non_aleatoric_binary", cfg.TEST.result,
              normalize=True)


if __name__=="__main__":
  assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
    'PyTorch>=0.4.0 is required'

  parser = argparse.ArgumentParser(description='Testing Modular '
                      'Segmentation Network Architectures for IVSLAM.')
  parser.add_argument(
                      "--cfg",
                      default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
                      metavar="FILE",
                      help="path to config file",
                      type=str)
  parser.add_argument(
                      "--gpus",
                      default="0",
                      help="gpus to use, e.g. 0-3 or 0,1,2,3")
  parser.add_argument(
                      "opts",
                      help="Modify config options using the command-line",
                      default=None,
                      nargs=argparse.REMAINDER)

  args = parser.parse_args()
  cfg.merge_from_file(args.cfg)
  cfg.merge_from_list(args.opts)

  print("Reading config file " + args.cfg)

  # Load the model
  if not cfg.TEST.is_ensemble:
    cfg.MODEL.weights_encoder = [cfg.TEST.test_model_encoder]
    cfg.MODEL.weights_decoder = [cfg.TEST.test_model_decoder]
    print("Loading encoder from " + cfg.MODEL.weights_encoder[0])
    print("Loading decoder from " + cfg.MODEL.weights_decoder[0])
    assert os.path.exists(cfg.MODEL.weights_encoder[0]) and \
          os.path.exists(
            cfg.MODEL.weights_decoder[0]), "checkpoint does not exitst!"
  else:
    print("Loading ensemble model: ")
    cfg.MODEL.weights_encoder = []
    cfg.MODEL.weights_decoder = []
    for encoder, decoder in zip(cfg.TEST.test_model_encoder_ensemble, 
                                cfg.TEST.test_model_decoder_ensemble):
      cfg.MODEL.weights_encoder += [encoder]
      cfg.MODEL.weights_decoder += [decoder]
      print("Loading encoder from " + encoder)
      print("Loading decoder from " + decoder)
      assert os.path.exists(encoder) and \
            os.path.exists(
              decoder), "checkpoint does not exitst!"
  

  # Parse gpu ids
  gpus = parse_devices(args.gpus)
  gpus = [x.replace('gpu', '') for x in gpus]
  gpus = [int(x) for x in gpus]
  num_gpus = len(gpus)

  main(cfg, args, gpus)

