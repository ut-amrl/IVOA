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

# Third party libs
from pixel_level_failure_detection.config import cfg
from networks.models import ModelBuilder, SegmentationModule
from lib.nn import patch_replication_callback
from lib.utils.utils import parse_devices
from pixel_level_failure_detection.lib.utils.utils import MaskedMSELoss

# TODO: 
# 1- update the dataset and use the additional config params
# 2- update the visualization/saving of the output images


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
  USE_MULTI_GPU = True if len(gpus) > 1 else False
  NUM_WORKERS = cfg.TEST.workers
  if cfg.TEST.use_gpu:
    BATCH_SIZE = cfg.TEST.batch_size_per_gpu * len(gpus)
  else:
    BATCH_SIZE = cfg.TEST.batch_size

  RESULT_SAVE_DIR = cfg.TEST.result + '/' + cfg.MODEL.name + '/'
  if not os.path.exists(RESULT_SAVE_DIR):
    os.makedirs(RESULT_SAVE_DIR)
  if not os.path.exists(RESULT_SAVE_DIR):
    print("Error: Could not create result directory ", RESULT_SAVE_DIR)
    exit()

  test_set_dict = {
    "test_tmp":[1007],
    "test_01_ganet_v0": [1007, 1012, 1017, 1022, 1027, 1032, 2007, 2012, 2017, 2022, 2027, 2032]
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

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  input_img_width = int(cfg.DATASET.img_width)
  input_img_height = int(cfg.DATASET.img_height)
  target_img_width = int(cfg.DATASET.img_width)
  target_img_height = int(cfg.DATASET.img_height)

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
  test_dataset = DepthErrorDataset(cfg.DATASET.root,
                                  cfg.DATASET.raw_img_root,
                                  session_list_test,
                                  loaded_image_color=cfg.DATASET.is_dataset_color,
                                  output_image_color=cfg.DATASET.use_color_images,
                                  session_prefix_length=cfg.DATASET.session_prefix_len,
                                  raw_img_folder=cfg.DATASET.raw_img_folder,
                                  label_img_folder=cfg.DATASET.label_img_folder,
                                  mask_img_folder=cfg.DATASET.mask_img_folder,
                                  transform_input=data_transform_input,
                                  transform_target=data_transform_target,
                                  load_masks=load_mask,
                                  regression_mode=cfg.MODEL.is_regression_mode,
                                  binarize_target=cfg.DATASET.binarize_target,
                                  no_meta_data_available=True,
                                  load_only_with_labels=cfg.TEST.ground_truth_available)
  datasets = {phases[0]: test_dataset}

  data_loaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS)
                  for x in phases}

  # Build the network from selected modules
  net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)
  net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    regression_mode=cfg.MODEL.is_regression_mode,
    inference_mode=True)

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
    criterion = nn.NLLLoss(ignore_index=-1)
    print("Classification Mode")

  if cfg.MODEL.arch_decoder.endswith('deepsup'):
    net = SegmentationModule(
      net_encoder, net_decoder, criterion, cfg.TRAIN.deep_sup_scale,
      segSize=desired_size)
  else:
    net = SegmentationModule(
      net_encoder, net_decoder, criterion, segSize=desired_size)

  # TODO: TEMPORARY Debugging
  param_size = 0
  for param in net.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in net.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()
  size_all_mb = (param_size + param_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))


  if cfg.TEST.use_gpu and USE_MULTI_GPU:
    if torch.cuda.device_count() >= len(gpus):
      available_gpu_count = torch.cuda.device_count()
      print("Using ", len(gpus), " GPUs out of available ", available_gpu_count)
      print("Used GPUs: ", gpus)
      net = nn.DataParallel(net, device_ids=gpus)
      # For synchronized batch normalization:
      patch_replication_callback(net)
    else:
      print("Requested GPUs not available: ", gpus)
      exit()

  net = net.to(device)

  print("Starting Inference...")
  start_time = time.time()

  timings_individual_all = []
  warm_up_iterations = 30
  if cfg.TEST.measure_inference_time:
    starter, ender = torch.cuda.Event(
    enable_timing=True), torch.cuda.Event(enable_timing=True)

  all_predictions = np.array([], dtype=np.int_)
  all_binary_labels = np.array([], dtype=np.int_)

  # Runs inference on all data
  for cur_phase in phases:
    # Set model to evaluate mode
    net.eval()

    running_loss = 0.0
    # Iterate over data
    for i, data in enumerate(tqdm(data_loaders[cur_phase]), 0):      
      # get the inputs
      input = data['img']
      img_names = data['img_name']
      session_nums = data['session']
      feed_dict = dict()
      feed_dict['input'] = input.to(device)
      

      mask_np = None
      if cfg.TRAIN.use_masked_loss:
        mask = data['mask_img']
        feed_dict['mask'] = mask.to(device)
        mask_np = mask.numpy()
      if not cfg.MODEL.is_regression_mode:
        target = data['labels']
        feed_dict['target'] = target.to(device)
      else:
        print("ERROR: Regression mode not supported")
        exit()



      # Do not track history since we are in eval mode
      with torch.set_grad_enabled(False):
        
        if cfg.TEST.measure_inference_time and i > warm_up_iterations:
          starter.record()
        
        # forward pass
        output = net(feed_dict)
        
        if cfg.TEST.measure_inference_time and i > warm_up_iterations:
          ender.record()
          torch.cuda.synchronize()
          timings_individual_all += [starter.elapsed_time(ender)]
        
        if cfg.TRAIN.use_masked_loss and cfg.MODEL.is_regression_mode:
          loss = criterion(output, feed_dict['target'], feed_dict['mask'])
        else:  
          loss = criterion(output, feed_dict['target'])

        if not cfg.MODEL.is_regression_mode:
          target = torch.reshape(target, (output.size()[0], 
                                          1,
                                          output.size()[2],
                                          output.size()[3]))
        
        prediction_label = torch.argmax(output, dim=1)

        
        if cfg.TRAIN.use_masked_loss:
          curr_labels = target[feed_dict['mask']].cpu().numpy().astype(np.int_)
          current_pred = prediction_label[torch.squeeze(feed_dict['mask'], 1)].cpu().numpy().astype(np.int_)
          
          all_predictions = np.concatenate((all_predictions, current_pred), 0) 
          all_binary_labels = np.concatenate((all_binary_labels, curr_labels), 0)
          
      ##TODO: TEMPORARY Debugging
      # mem_c = torch.cuda.memory_allocated()
      # print("Current Memory allocated (MB): ", mem_c * 1e-6)


      input_np = input.to(torch.device("cpu")).numpy()
      target_np = target.to(torch.device("cpu")).numpy()
      output_np = output.to(torch.device("cpu")).numpy()

      # TODO: Update and enable save images
      # save_result_images(input_np,
      #                    target_np,
      #                    output_np,
      #                    img_names,
      #                    session_nums,
      #                    RESULT_SAVE_DIR,
      #                    raw_output_size = raw_output_img_size,
      #                    gt_available=cfg.TEST.ground_truth_available,
      #                    save_raw_output=cfg.TEST.save_raw_output,
      #                    initial_directory_prep=(i == 0),
      #                    gt_masks = mask_np)

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
  
  cnf_matrix_binary = confusion_matrix(all_binary_labels,
                                       all_predictions)
  plot_confusion_matrix(
            cnf_matrix_binary,
            ['NF', 'F'],
            "confusion_mat_binary", cfg.TEST.result,
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
  cfg.MODEL.weights_encoder = cfg.TEST.test_model_encoder
  cfg.MODEL.weights_decoder = cfg.TEST.test_model_decoder
  print("Loading encoder from " + cfg.MODEL.weights_encoder)
  print("Loading decoder from " + cfg.MODEL.weights_decoder)
  assert os.path.exists(cfg.MODEL.weights_encoder) and \
         os.path.exists(
           cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

  # Parse gpu ids
  gpus = parse_devices(args.gpus)
  gpus = [x.replace('gpu', '') for x in gpus]
  gpus = [int(x) for x in gpus]
  num_gpus = len(gpus)

  main(cfg, args, gpus)

