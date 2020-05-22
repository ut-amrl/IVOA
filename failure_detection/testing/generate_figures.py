# Copyright 2020 srabiee@cs.umass.edu
# Department of Computer Sciences,
# University of Texas at Austin


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

# This script loads a provided input image as well as IVOAs probability scores
# of FP and FN for the same image (outputs of the run_evaluation.py). It then
# overlays the FP+FN error heatmap on the original image.

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib
import cv2


def main():
  # If set to true, only regions in the image that have been predicted to cause
  # errors for percpetion with high probability will be highlighted. If set to
  # false the complete predicted failure probability heatmap will be overlaid on
  # the input image
  THRESHOLD_FAILURE = True
  THRESHOLD_VAL = 0.75
  ADD_ON_GRAY_IMAGE = False


  # input_img_path = "/media/ssd2/datasets/AirSim_IVOA/CRA_ML_tool/0002_01_418sec_daytime_leaves100/img_left/0000000325.png"
  # fp_img_path = "/media/ssd2/results/IVOA/apr_07_20/processed_imgs/00002/perception_false_pos/00002_0000000325_l.txt"
  # fn_img_path = "/media/ssd2/results/IVOA/apr_07_20/processed_imgs/00002/perception_false_neg/00002_0000000325_l.txt"

  # input_img_path = "/media/ssd2/datasets/AirSim_IVOA/CRA_ML_tool/0002_01_418sec_daytime_leaves100/img_left/0000000183.png"
  # fp_img_path = "/media/ssd2/results/IVOA/apr_07_20/processed_imgs/00002/perception_false_pos/00002_0000000183_l.txt"
  # fn_img_path = "/media/ssd2/results/IVOA/apr_07_20/processed_imgs/00002/perception_false_neg/00002_0000000183_l.txt"

  input_img_path = "/media/ssd2/datasets/AirSim_IVOA/CRA_ML_tool/0001_01_418sec_daytime_fog100/img_left/0000000040.png"
  fp_img_path = "/media/ssd2/results/IVOA/apr_07_20/processed_imgs/00001/perception_false_pos/00001_0000000040_l.txt"
  fn_img_path = "/media/ssd2/results/IVOA/apr_07_20/processed_imgs/00001/perception_false_neg/00001_0000000040_l.txt"


  input_img = io.imread(input_img_path)
  width = input_img.shape[1]
  height = input_img.shape[0]
  input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
  # cv2.imwrite("input_img_gray.png", input_img_gray)

  fp_img = np.loadtxt(fp_img_path, dtype=float)
  fn_img = np.loadtxt(fp_img_path, dtype=float)

  failure_img = fp_img + fn_img
  failure_img = failure_img / np.amax(failure_img)
  failure_img = cv2.resize(failure_img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

  if THRESHOLD_FAILURE:
    remove_idx = np.nonzero(failure_img < THRESHOLD_VAL)
    failure_img[remove_idx] = 0
    keep_idx = np.nonzero(failure_img >= THRESHOLD_VAL)
    failure_img[keep_idx] = 1

  # Prepare the error heatmap
  if THRESHOLD_FAILURE:
    cmap = 'bwr'
  else:
    cmap = 'viridis'

  interp = 'bilinear'
  fig = plt.figure(frameon=False)
  fig.set_size_inches(4 * width/height, 4 * 1.0)
  # fig.set_size_inches(4 * 1.6, 4 * 1.0)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(failure_img, cmap=cmap, interpolation=interp,
                       vmin=0.0, vmax=1.0, aspect='auto')
  fig.savefig("error_heatmap.png", dpi=150)

  loaded_heatmap = io.imread("error_heatmap.png")

  # Blend the heatmap with the original image
  if ADD_ON_GRAY_IMAGE:
    input_img_3C = cv2.cvtColor(input_img_gray, cv2.COLOR_GRAY2BGRA)
  else:
    input_img_3C = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGRA)
  loaded_heatmap = cv2.cvtColor(loaded_heatmap, cv2.COLOR_RGBA2BGRA)
  alpha = 0.5

  # Blend only the pixels in the heatmap that are classified as error
  if THRESHOLD_FAILURE:
    for i in range(loaded_heatmap.shape[0]):
      for j in range(loaded_heatmap.shape[1]):
        if failure_img[i, j]:
          input_img_3C[i, j, 0:3] = (input_img_3C[i, j, 0:3] * alpha
                                     + (1-alpha) * loaded_heatmap[i, j, 0:3])
  else:
    cv2.addWeighted(loaded_heatmap, alpha, input_img_3C, 1 - alpha,
                    0, input_img_3C)
  cv2.imwrite("overlay.png", input_img_3C)


if __name__=="__main__":
  main()