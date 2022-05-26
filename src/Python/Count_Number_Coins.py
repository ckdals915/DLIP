"""
# DLIP - Python OpenCV Tutorial
Created by Smart Sensor System Lab
2022-1
> mod: 2022.4.18
This is OpenCV Tutorial for Python (*.py) Version. 
Opens and shows Webcam images
"""



"""## Import OpenCV Library"""

import math
import numpy as np
import cv2 as cv
from cv2 import *
from cv2.cv2 import *
from matplotlib import pyplot as plt

# Load image (BGR2GRAY)
img = cv.imread('coins_noisy.jpg')
gray_img = cvtColor(img, COLOR_BGR2GRAY)


# # Plot Histogram
# histSize = 256
# histRange = (0, 256)
# b_hist = cv.calcHist(img, [0], None, [histSize], histRange, False)

# hist_w = 512
# hist_h = 400
# bin_w = int(round(hist_w/histSize))
# histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

# cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
# for i in range(1, histSize):
#     cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
#             ( bin_w*(i), hist_h - int(b_hist[i]) ),
#             ( 255, 0, 0), thickness=2)

# # Plot Results
# plt.subplot(2,2,1),plt.imshow(img, 'gray')
# plt.title('Source image')
# plt.xticks([]),plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(histImage)
# plt.title('CalcHist image')
# plt.xticks([]),plt.yticks([])
# plt.show()

# Apply a filter to remove image noises
filtered_img = cv.medianBlur(gray_img, 7)

# Choose the appropriate threshold value
thVal = 127
_,thresh_img = threshold(filtered_img, thVal, 255, cv.THRESH_BINARY)
plt.imshow(thresh_img, "gray")
plt.title("Threshold image")
plt.xticks([]),plt.yticks([])
plt.show()

# Apply the appropriate morphology method to segment coins
getStructuringElement(MORPH_RECT, (5,5))
kernel = np.ones((5,5), np.uint8)


# Find the contour and draw the segmented objects

# Exclude the contours which are too small or too big

# Count the number of each different coins

# Calculate the total amount of money

