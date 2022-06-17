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
img = imread('FinalTest_Q3_image1.jpg')


# Show Image using colab imshow
cv.imshow('source',img)
cv.WaitKey(0)