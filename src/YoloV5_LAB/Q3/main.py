'''
* *****************************************************************************
* @author	ChangMin An
* @Mod		2022 - 06 - 03
* @brief	Final
* @Version	Python 3.9.12, CUDA 11.3.1(RTX 3060 Laptop), pytorch 1.10 
* *****************************************************************************
'''

#===============================================#
#              Open Library Declare             #
#===============================================#
import torch
import cv2 as cv
from matplotlib import pyplot as plt
from cv2 import *
from cv2.cv2 import *
import random
from PIL import Image
import numpy as np

#===============================================#
#                Global Variable                #
#===============================================#
# Video Variable
# src = imread('FinalTest_Q3_image1.jpg')
src = imread('FinalTest_Q3_image2.jpg')

# Color Definition (BGR)
WHITE               = (255, 255, 255)
RED                 = (  0,   0, 255)
GREEN               = (  0, 255,   0)
PINK                = (184,  53, 255)
YELLOW              = (  0, 255, 255)
BLUE                = (255,   0,   0)
BLACK               = (  0,   0,   0)
PURPLE              = (255, 102, 102)

# Font Definition
USER_FONT           = FONT_HERSHEY_DUPLEX
TRANSPARENCY        = 0.3

people_Count       = 0
#===============================================#
#                     Main                      #
#===============================================#

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=True)

# Threshold Configuration
# model.conf = 0.1

# Run Inference
results = model(src)

# Calculate the Bounding Box result using Pandas
L_xyxy = len(results.pandas().xyxy[0])
for i in range(L_xyxy):
  if results.pandas().xyxy[0].name[i] == "person":
    people_Count += 1

# Count Text
Count_Text = f"Number of People: {people_Count}"
putText(src, Count_Text, (50, 50), USER_FONT, 1, BLACK)

imshow("Source", src)
waitKey(0)
