'''
* *****************************************************************************
* @author	ChangMin An
* @Mod		2022 - 05 - 24
* @brief	DLIP LAB: Parking Management System
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
Video = "DLIP_parking_test_video.avi"

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

#===============================================#
#                     Main                      #
#===============================================#

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=True)

# Threshold Configuration
# model.conf = 0.6 # Threshold가 바뀌는 것!!
# model.iou = 0.5 # iou 값을 바꾸는것

# Open the Video
cap = cv.VideoCapture(Video)
if (cap.isOpened() == False):
  print("Not Open the VIDEO")

#================== While Loop =================#    
while(1):
  # Start Window Time
  startTime = cv.getTickCount()

  # Read Video Capture
  cap_Flag, src = cap.read()
  
  src_gray = cvtColor(src, cv.COLOR_BGR2GRAY)

  # Pre-Processing
  src_filtered = GaussianBlur(src_gray, (5,5), 0)

  # If Capture is failed, break the loop
  if cap_Flag == False:
    print("Video End")
    break
  
  # ROI Setting
  roi = np.zeros_like(src_filtered)
  src_roi = np.zeros_like(src_filtered)
  roi_point = np.array([[60,280], [1280,280], [1280,430], [0, 440]])
  roi = fillConvexPoly(roi, roi_point, WHITE)
  bitwise_and(src_gray, src_gray, src_roi, roi)

  # bounding box 크기에 따라 없애기

  # 겹치는 bounding box 없애기

  # Run Inference
  results = model(src_roi)

  # Print Results
  # results.print()

  # # Save Result images with bounding box drawn
  # results.save()  # or .show()
  

  # Print the Bounding Box result using Pandas
  L_xyxy = len(results.pandas().xyxy[0])
  for i in range(L_xyxy):
    Xmin = round(results.pandas().xyxy[0].xmin[i])
    Xmax = round(results.pandas().xyxy[0].xmax[i])
    Ymin = round(results.pandas().xyxy[0].ymin[i])
    Ymax = round(results.pandas().xyxy[0].ymax[i])

    if results.pandas().xyxy[0].name[i] == "car":
      cv.rectangle(src, (Xmin, Ymin), (Xmax, Ymax), RED, 2)

  


  #print(results.pandas().xyxy[0],'\n')  # imgs predictions (pandas)

  # Press Esc to Exit, Stop Video to 's' 
  k = cv.waitKey(5) & 0xFF
  if k == 27:
    break
  elif k == ord('s'):
    cv.waitKey()
  
  # Time Loop End
  endTime = cv.getTickCount()

  # FPS Calculate
  FPS = round(getTickFrequency()/(endTime - startTime))

  # FPS Text
  FPS_Text = f"FPS: {FPS}"
  putText(src, FPS_Text, (100, 160), USER_FONT, 1, BLUE)

  # Count Text
  Count_Text = f"Number of Car: {L_xyxy}"
  putText(src, Count_Text, (100, 100), USER_FONT, 1, BLUE)

  # # Show result image
  # cv.imshow("result", (results.imgs[randNo])[:,:,::-1])
  imshow("source", src)


# Release
cv.destroyAllWindows()
cap.release()