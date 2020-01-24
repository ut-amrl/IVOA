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

import torch
import torchvision.models as models
import torch.nn as nn
import argparse, os

# This network structure is based on alexnet and performs classification of
# 4 different classes: JPP estimate being 1- true positive, 2-true negative
#                      3- false positive, and 4- false negative. Positive means
#                      estimating that there exists and obstacle and negative
#                      means estimating that there exists no obstacles.
class FailureDetectionMultiClassNet(nn.Module):
    
    def __init__(self, base_model="alexnet", lock_feature_ext=True):
        super(FailureDetectionMultiClassNet, self).__init__()
        
        supported_models = ["alexnet", "resnet152", "inception_v3"]
        if not(base_model in supported_models):
            print(base_model, " is not supported. Available models"
                  ," include: ", supported_models)
        
        if base_model == "alexnet":
            model = models.alexnet(pretrained=True)
          
            ## Number of filters in the bottleneck layer
            #num_ftrs = model.classifier[6].in_features
            ## convert all the layers to list and remove the last one
            #features = list(model.classifier.children())[:-1]
            ### Add the last layer based on the num of classes in our dataset
            #features.extend([nn.Linear(num_ftrs, 2)])
            ### convert it into container and add it to our model class.
            #model.classifier = nn.Sequential(*features)
            
            # Freezes all parameters
            if (lock_feature_ext):
                for name, param in zip(model.state_dict(), model.parameters()):
                    param.requires_grad = False
            
            # Replaces the fully connected layers (These new parameters will
            # be tunable)
            model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 4),
                )
        elif base_model == "resnet152":
            model = models.resnet152(pretrained=True)
            
            # Freezes all parameters
            if (lock_feature_ext):
                for name, param in zip(model.state_dict(), model.parameters()):
                    param.requires_grad = False
            
            # Replaces the fully connected layers (These new parameters will
            # be tunable)
            model.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 4, 1500),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1500, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 4),
                )
            
        elif base_model == "inception_v3":
            model = models.inception_v3(pretrained=True)
            
            # Freezes all parameters
            if (lock_feature_ext):
                for name, param in zip(model.state_dict(), model.parameters()):
                    param.requires_grad = False
            
            #Replaces the fully connected layers (These new parameters will
            #be tunable)
            model.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(2048, 1500),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1500, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 4),
                )
      
        
        self.net = model
        
    def forward(self, input):
        return self.net(input)
    
if __name__=="__main__":
    net = FailureDetectionMultiClassNet(base_model="alexnet")
   
