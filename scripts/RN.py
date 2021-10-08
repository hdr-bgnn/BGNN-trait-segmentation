import pandas as pd
import os
import torch
import torch.nn.functional as F
import numpy as np


class RN(torch.nn.Module):
    def __init__(self):
        super(RN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(4, 64, 7, stride=3, padding=3)
        self.batchNorm1 = torch.nn.BatchNorm2d(64)
        
        self.conv2 = torch.nn.Conv2d(64, 24, 7, stride=3, padding=3)
        self.batchNorm2 = torch.nn.BatchNorm2d(24)
        
        self.conv3 = torch.nn.Conv2d(24, 24, 7, stride=3, padding=3)
        self.batchNorm3 = torch.nn.BatchNorm2d(24)
        

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(24*12*30, 100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x