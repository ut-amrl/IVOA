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

import sys
import numpy as np
import time
import json
import copy
from PIL import Image, ImageDraw, ImageFont
import argparse, os
from skimage import io, transform
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from math import floor, ceil
from collections import OrderedDict
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages


def make_bbox(point, size):
    bbox = [(point[0] - round(size[0]/2), 
             point[1] - round(size[1]/2)),
            (point[0] + round(size[0]/2), 
             point[1] + round(size[1]/2))]
    return bbox

# The query points are np.array(n*2) where the first column is the height (y) 
# and the second column is the width (x) for the queried pixel
def annotate_results_on_image(raw_image, query_pts, ground_truth, predictions,
                              classes, save_path, img_name):
    color_list = ['green', 'blue', 'gold', 'red']
    gt_circle_rad = 10
    pred_circle_rad = 5
    
    legend_pos= np.array([40, 20], dtype=int)
    legend_vspace = 50
    legend_hspace = 50
    legend_box_size = np.array([15, 15])
    legend_text_size = np.array([15, 15])
    #font = ImageFont.truetype("arial.ttf", 15)
    
    img = raw_image.copy()
    img_draw = ImageDraw.Draw(img)
    
    for i in range(ground_truth.size):
        bbox_pred = [(query_pts[i,1] - pred_circle_rad, 
                      query_pts[i,0] - pred_circle_rad),
                      (query_pts[i,1] + pred_circle_rad,
                      query_pts[i,0] + pred_circle_rad)]
        
        bbox_gt = [(query_pts[i,1] - gt_circle_rad, 
                    query_pts[i,0] - gt_circle_rad),
                    (query_pts[i,1] + gt_circle_rad,
                    query_pts[i,0] + gt_circle_rad)]
                    
        img_draw.ellipse(bbox_gt, fill=color_list[ground_truth[i]])
        img_draw.ellipse(bbox_pred, fill=color_list[predictions[i]])
        
    # Draw the legend
    
    for i in range(len(classes)):
        pos = legend_pos
        pos[1] = pos[1] + legend_vspace
        bbox = make_bbox(pos, legend_box_size)
        img_draw.rectangle(bbox, fill=color_list[i])
        
        text_pos = np.copy(pos)
        text_pos[0] = text_pos[0] + legend_hspace
        text_pos = text_pos - np.round(legend_text_size/2)
        img_draw.text(text_pos.tolist(), classes[i], fill=color_list[i])
        
    
    out_img = Image.blend(img, raw_image, 0.1)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory " , save_path ,  " Created ")

    out_img.save(save_path + '/' + img_name, format='JPEG')



    
