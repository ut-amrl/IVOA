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

"""
This is similar to test_multi_class_model with the difference that input data are loaded in the form of individual images and then image patches are extracted and passed to the model. This is done to better test the performance of the model in a real-time application scenario and also for the purpose of saving visualizations of theresults.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

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
from data_loader.load_full_images import FailureDetectionDataset
from networks.failure_detection_multi_class import FailureDetectionMultiClassNet
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import time


def append_dummy_patches(inputs,
                         full_img_shape, patch_crop_size, patch_stride,  patch_extraction_margin):
    img_height = full_img_shape[1]
    img_width = full_img_shape[2]

    query_samples_x = floor((img_width - 2 * patch_extraction_margin - patch_crop_size) / patch_stride + 1 )
    query_samples_y = floor((img_height - 2 * patch_extraction_margin - patch_crop_size) / patch_stride + 1)
    full_img_query_point_size =  query_samples_x * query_samples_y

    current_patch_num = inputs.shape[0]
    dummy_patch_num = full_img_query_point_size - current_patch_num
    dummy_patches = torch.zeros(dummy_patch_num, inputs.shape[1], inputs.shape[2], inputs.shape[3])
    inputs = torch.cat((inputs, dummy_patches), 0)
    return inputs


def convert_image_name_to_index(image_name, session_name):
    """
    Converts the image name to an index.
    :param image_name
    :param session_name
    :return: index
    """
    image_index = image_name.replace(
        "_l.jpg", "").lstrip(session_name).lstrip("_")

    return image_index

def visualize_patch_failure_predictions(patch_failure_pred_labels,
                                        gt_patch_multi_class_labels,
                                        patch_coords,
                                        rgb_image,
                                        output_folder_path,
                                        image_idx):
    """
    Draws a visualization of the predicted and ground truth labels for a patch on 
    the original image. The predicted labels are drawn as circles on the center 
    of the patch. The ground truth labels are drawn as outer rings around the 
    prediction.
    :return:
    """
    failure_color = (0, 0, 255)  # red (BGR)
    non_failure_color = (0, 255, 0)  # green (BGR)

    prediction_circle_radius = 2  # 5, 2
    gt_circle_radius = 5  # 10, 5
    thickness = -1

    gt_binary_labels = np.logical_or(
        gt_patch_multi_class_labels == 2, gt_patch_multi_class_labels == 3)

    i = 0
    image = rgb_image.copy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) * 255.0
    image = np.ascontiguousarray(image, dtype=np.uint8)
    for i in range(patch_coords.shape[0]):
        gt_color = non_failure_color if gt_binary_labels[i] == 0 else failure_color
        pred_color = non_failure_color if patch_failure_pred_labels[i] == 0 else failure_color

        center_coordinates = (patch_coords[i, 1], patch_coords[i, 0])
        image = cv2.circle(image, center_coordinates,
                    gt_circle_radius, gt_color, thickness)
        image = cv2.circle(image, center_coordinates,
                    prediction_circle_radius, pred_color, thickness)
        i += 1

    # Create output folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_image_path = os.path.join(output_folder_path, image_idx + ".png")
    cv2.imwrite(output_image_path,
                image)


def extract_patches_from_img(full_img,
                             patch_coords_left, patch_size, transform):
    """
    Extracts patches from the full image given the coordinates of the patches and applies the provided transform to the image patches.
    """
    patch_coords = patch_coords_left.numpy()
    half_patch_size = floor(patch_size / 2)
    patches_transformed = torch.tensor([], dtype=torch.float)

    i = 0
    for i in range(patch_coords.shape[0]):
        patch_coord = patch_coords[i, :]
        patch_x_start = floor(
            patch_coord[1] - half_patch_size)
        patch_y_start = floor(
            patch_coord[0] - half_patch_size)
        patch_x_end = floor(patch_x_start + patch_size)
        patch_y_end = floor(patch_y_start + patch_size)

        assert patch_x_start >= 0 and patch_y_start >= 0 and patch_x_end < full_img.shape[
            2] and patch_y_end < full_img.shape[1], "Error: patch coordinates are out of bounds. Patch coords: {}:{}, {}:{} \n".format(patch_x_start, patch_x_end, patch_y_start, patch_y_end) + "patch coords: {} \n \n".format(patch_coords)

        patch = full_img[:,patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        patch_transformed = transform(patch)
        patch_transformed = torch.reshape(patch_transformed, (1, patch_transformed.shape[0], patch_transformed.shape[1], patch_transformed.shape[2]))
        patches_transformed = torch.cat((patches_transformed, patch_transformed), 0)
        i += 1

    return patches_transformed

# Helper function for saving inference results
def save_results(target_dir, file_name, classes,
                 ground_truth, predictions, class_prob, class_uncertainty):

    # Handle the case when there is no class_uncertainty available
    formatted_class_uncertainty = {}
    if(class_uncertainty.size != 0):
        formatted_class_uncertainty = {
            classes[x]: class_uncertainty[:, x].tolist()
            for x in range(len(classes))}
    formatted_class_prob = {}
    if(class_prob.size != 0):
        formatted_class_prob = {
            classes[x]: class_prob[:, x].tolist()
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
    file = open(target_dir + '/' + file_name + '_data.json', 'w')
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


def main():
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
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                        help='Is the model trained using multiple GPUs',
                        required=True)
    parser.add_argument('--use_gpu', default=None,
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                        help='If you want to use GPUs.',
                        required=True)
    parser.add_argument('--use_color_images', default=None,
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                        help='True if you want to feed the network with color images',
                        required=True)
    parser.add_argument('--calc_uncertainty', default=None,
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                        help='True if you want to calculate uncertainty values',
                        required=True)
    parser.add_argument('--online_patch_extraction', default=None,
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                        help='True if you want to extract patches from full images',
                        required=True)
    parser.add_argument('--patch_crop_size', default=36,
                        help=('The size the of original patch before being '
                              'resized to fit the network input size.'),
                        type=int, required=True)
    parser.add_argument('--patch_stride', default=15,
                        help=('The stride size in pixels between two patches.'),
                        type=int, required=True)
    parser.add_argument('--patch_extraction_margin', default=20,
                        help=('The margin size in the edges of the images to throw out during patch extraction. (pixels)'),
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

    PARTICLE_NUM = 50  # number of passes for each input image def:50
    DATASET_IMAGE_CHANNEL = 3  # number of channels of the dataset
    LOAD_MODEL_WEIGHTS = True
    USE_MULTI_GPU = True
    NUM_WORKERS = 16 
    BATCH_SIZE = 1  
    MEASURE_INFERENCE_TIME = True
    SAVE_PREDICTION_VISUALIZATION = True

    test_set_dict = {
        "test_1": [1, 2, 3, 4, 5],
        "test_1_ganet_v0": [1007],
        "test_01_ganet_v0": [1007, 1012, 1017, 1022, 1027, 1032, 2007, 2012, 2017, 2022, 2027, 2032]
    }

    if (TEST_SET_NAME is not None):
        bagfile_list_test = test_set_dict[TEST_SET_NAME]
        RESULT_FILE_NAME = RESULT_FILE_NAME + "_" + TEST_SET_NAME

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
        total_mem = (
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
        patch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        full_img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        patch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        full_img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ])


    output_image_channel = 3 if USE_COLOR_IMAGES else 1
    test_dataset = FailureDetectionDataset(root_dir,
                                           bagfile_list_test,
                                           DATASET_IMAGE_CHANNEL,
                                           output_image_channel,
                                           full_img_transform,
                                           meta_data_dir=META_DATA_DIR,
                                           load_patch_info = True)

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
            name = k[7:]  # remove 'module.'
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

    if MEASURE_INFERENCE_TIME:
        starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)

        # Since the returned patches for each image are limited to those 
        # that are within max range limits, we append dummy patches to the
        # end of each batch just for the sake of inference time measurement.
        APPEND_DUMMY_PATCHES = True
    else:
        APPEND_DUMMY_PATCHES = False

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

        # Samples of inference time of a single model in the ensemble (a single pass of data through the model if running in MC mode) on batch_size number of datapoints
        timings_individual_all = []
        # Inference time of each of the models in the ensemble on the latest data batch (all image patches in a single image)
        timings_individual_per_batch = []
        # Samples of the total inference time of all models in the ensemble on batch_size number of datapoints (Sum of all passes of the same data through the network in the MC dropout mode)
        timings_ensemble_all = []
        # Iterations to run before starting timing
        warm_up_iterations = 20

        # Iterate over data
        for i, data in enumerate(data_loaders[cur_phase], 0):
            timings_individual_per_batch = []
            #class_count += count_class_instances(data)
            #print('Class count: ', class_count)

            # get the inputs
            full_img = data['full_img']
            img_name = data['full_img_name']
            session_name = data['bagfile_name'][0]
            patch_coords_left = data['patch_coords_left']
            multi_class_labels = data['multi_class_label']
            img_idx = convert_image_name_to_index(
                img_name[0], session_name)
            inputs = extract_patches_from_img(full_img[0],
                             patch_coords_left[0], PATCH_SIZE, patch_transform)
            patch_num = patch_coords_left.shape[1]
            

            # Extract additional dummy patches such that the list covers all the input image. Used for timing measurements
            if APPEND_DUMMY_PATCHES:
                inputs = append_dummy_patches(inputs, full_img[0].shape, args.patch_crop_size,  args.patch_stride, args.patch_extraction_margin)

            # TODO: Should we extract additional dummy patches such that the list covers all the input image?

            inputs = inputs.to(device)
            labels = torch.tensor(multi_class_labels, dtype=torch.int, device=device)

            print("inputs shape: ", inputs.shape)

            particle_class_probs = torch.tensor([], dtype=torch.float,
                                                device=device)
            
            # ++++++
            #---------------------- 
            # Pass the same patch multiple times through the network if
            # uncertainty calculation is required
            # TODO: Add support for inference time measurement for MC dropout mode
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

                all_class_probs = torch.cat(
                    (all_class_probs, mean_class_prob), 0)
                all_class_var = torch.cat(
                    (all_class_var, predicted_class_var), 0)
            
            #---------------------- 
            # Normal operation with no uncertainty calculation (it is
            # separated from the CALC_UNCERTAINTY mode for runtime efficiency)
            else:
                # track history if only in train
                with torch.set_grad_enabled(False):
                    # Time the network latency
                    if MEASURE_INFERENCE_TIME and i > warm_up_iterations:
                        starter.record()
                    outputs = net(inputs)
                    if MEASURE_INFERENCE_TIME and i > warm_up_iterations:
                        ender.record()
                        torch.cuda.synchronize()
                        timings_individual_per_batch += [starter.elapsed_time(ender)]
                        timings_individual_all += [starter.elapsed_time(ender)]

                    _, mean_preds = torch.max(outputs, 1)
                    mean_preds = mean_preds[0:patch_num]

                relevant_class_probs = softmax_layer(outputs)[0:patch_num,:]
                all_class_probs = torch.cat((all_class_probs,
                                            relevant_class_probs), 0)

            # ++++++

            # TODO: convert this to np and push to cpu before concatenation
            all_predictions = torch.cat((all_predictions, mean_preds), 0)
            all_multi_class_labels = np.append(
                all_multi_class_labels, np.array(multi_class_labels, dtype=int))

            corrects = torch.sum(mean_preds == labels)
            running_corrects += corrects

            if i % checkpoint_num == checkpoint_num - 1:  # print every
                # checkpoint_num 1000 mini-batches
                print('%d%%: Processed %d datapoints.'
                      % (((i + 1) * BATCH_SIZE * 100.00) / dataset_size,
                         (i + 1) * BATCH_SIZE))
                if used_gpu_count:
                    print("Total GPU usage (MB): ",
                          calculate_gpu_usage(used_gpu_count), " / ",
                          used_gpu_count * total_mem)

            if MEASURE_INFERENCE_TIME and i > warm_up_iterations:
                timings_ensemble_all += [sum(timings_individual_per_batch)]
                mean_inf_time = np.mean(np.array(timings_individual_per_batch))
                std_inf_time = np.std(np.array(timings_individual_per_batch))
                print("Mean inference time across the ensemble models on current batch: {:.4f} ms, Std: {:.4f} ms".format(
                    mean_inf_time, std_inf_time))             

            if SAVE_PREDICTION_VISUALIZATION:
                visualize_patch_failure_predictions(
                    patch_failure_pred_labels=mean_preds.cpu().numpy(),
                    gt_patch_multi_class_labels=np.array(
                        multi_class_labels, dtype=int),
                    patch_coords=patch_coords_left[0].numpy(),
                    rgb_image=full_img[0].numpy(),
                    output_folder_path=os.path.join(args.save_dir,
                                                    session_name, 'failure_pred_patch_vis'),
                    image_idx=img_idx)

        all_predictions_np = all_predictions.to(torch.device("cpu")).numpy()
        all_class_probs_np = all_class_probs.to(torch.device("cpu")).numpy()
        all_class_var_np = all_class_var.to(torch.device("cpu")).numpy()
        print('All data was processed.')
        time_elapsed = time.time() - start_time
        print('Completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
        print('Acc: %f' % accuracy_score(all_multi_class_labels,
                                         all_predictions_np,
                                         normalize=True))
        cnf_matrix = confusion_matrix(all_multi_class_labels,
                                      all_predictions_np)

        if MEASURE_INFERENCE_TIME:
            mean_inf_time = np.mean(np.array(timings_individual_all))
            std_inf_time = np.std(np.array(timings_individual_all))
            print("Mean inference time for individual models: {:.4f} ms, Std: {:.4f} ms".format(
                mean_inf_time, std_inf_time))

            mean_inf_time = np.mean(np.array(timings_ensemble_all))
            std_inf_time = np.std(np.array(timings_ensemble_all))
            print("Mean inference time for the ensemble: {:.4f} ms, Std: {:.4f} ms".format(
                mean_inf_time, std_inf_time))

        # TODO: remove debugging
        print("all_patch_multi_class_labels.shape: ",
              all_multi_class_labels.shape)

        # Convert the multi-class ground truth labesl to binary labels (FP and FN -> Failure, TP and TN -> No failure)
        # Failure (1) and No failure (0)
        all_binary_labels = np.logical_or(
            all_multi_class_labels == 2, all_multi_class_labels == 3)
        all_binary_predictions = np.logical_or(
            all_predictions_np == 2, all_predictions_np == 3)

        cnf_matrix_binary = confusion_matrix(all_binary_labels,
                                             all_binary_predictions)

        unique_class_labels = np.unique(all_multi_class_labels)
        unique_class_label_predictions = np.unique(all_predictions_np)
        if len(unique_class_labels) != len(unique_class_label_predictions):
            print("WARNING: Number of unique ground truth class labels and  unique class predictions are different")
            print("unique_class_labels:", unique_class_labels)
            print("unique_class_label_predictions: ",
                  unique_class_label_predictions)

        class_names = ['TP', 'TN', 'FP', 'FN']
        available_class_names = [
            class_names[int(i)] for i in unique_class_labels]

        plot_confusion_matrix(
            cnf_matrix,
            available_class_names,
            RESULT_FILE_NAME, RESULTS_SAVE_DIR,
            normalize=True)

        plot_confusion_matrix(
            cnf_matrix_binary,
            ['NF', 'F'],
            RESULT_FILE_NAME + "_binary", RESULTS_SAVE_DIR,
            normalize=True)

        save_results(RESULTS_SAVE_DIR, RESULT_FILE_NAME,
                     class_names,
                     all_multi_class_labels, all_predictions_np,
                     all_class_probs_np, all_class_var_np)


if __name__ == "__main__":
    main()
