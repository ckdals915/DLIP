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
# Video Variable
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
TRANSPARENCY        = 0.3

# Variable about Parking Point 
center_Point = []
parking_Point = [(78,378),(175,380),(289,379),(384,379),(490,378),(580,378),(690,378),(774,376),(885,376),(977,373),(1084,373),(1179,373),(1260,336)]
parking_Coordinate = [[(70,325),(175,325),(85,430),(0,430)],        [(175,325),(265,325),(200,430),(85,430)],     [(265,325),(360,325),(312,430),(200,430)],
                      [(360,325),(454,325),(423,430),(313,430)],    [(454,325),(546,325),(527,430),(423,430)],    [(546,325),(635,325),(635,430),(527,430)],
                      [(635,325),(720,325),(747,430),(635,430)],    [(720,325),(810,325),(850,430),(747,430)],    [(810,325),(900,325),(956,430),(850,430)],
                      [(900,325),(990,325),(1065,430),(956,430)],   [(990,325),(1080,325),(1175,430),(1065,430)], [(1080,325),(1165,325),(1280,430),(1175,430)],
                      [(1165,325),(1280,300),(1280,400),(1280,430)]]
parking_Flag = [False,False,False,False,False,False,False,False,False,False,False,False,False]

# Frame Text Variable
frame_Num = 0

# Open the File
f = open("counting_result.txt",'w')
f.write("Frame\tNumber of Car\n")

#===============================================#
#                     Main                      #
#===============================================#

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=True)

# Threshold Configuration
model.conf = 0.1

# Open the Video & Recording Video Configuration
cap = cv.VideoCapture(Video)
w = round(cap.get(CAP_PROP_FRAME_WIDTH))
h = round(cap.get(CAP_PROP_FRAME_HEIGHT))
fps = cap.get(CAP_PROP_FPS)
fourcc = VideoWriter_fourcc(*'DIVX')
out = VideoWriter('output.avi', fourcc, fps, (w,h))
delay = round(1000/fps)

if (cap.isOpened() == False):
  print("Not Open the VIDEO")

#================== While Loop =================#    
while(1):
  # Start Window Time
  startTime = cv.getTickCount()

  # Initialization
  car_Count = 0
  parking_Flag = [False,False,False,False,False,False,False,False,False,False,False,False,False]
  
  # Read Video Capture
  cap_Flag, src = cap.read()

  # If Capture is failed, break the loop
  if cap_Flag == False:
    print("Video End")
    break
  
  # Pre-Processing
  src_gray = cvtColor(src, cv.COLOR_BGR2GRAY)
  src_filtered = GaussianBlur(src_gray, (5,5), 0)
  
  # ROI Setting
  roi = np.zeros_like(src_filtered)
  src_roi = np.zeros_like(src_filtered)
  roi_point = np.array([[60,280], [1280,280], [1280,430], [0, 440]])
  roi = fillConvexPoly(roi, roi_point, WHITE)
  bitwise_and(src_filtered, src_filtered, src_roi, roi)

  # Run Inference
  results = model(src_roi)

  # Calculate the Bounding Box result using Pandas
  L_xyxy = len(results.pandas().xyxy[0])
  for i in range(L_xyxy):
    # X, Y Coordinate Value
    Xmin = round(results.pandas().xyxy[0].xmin[i])
    Xmax = round(results.pandas().xyxy[0].xmax[i])
    Ymin = round(results.pandas().xyxy[0].ymin[i])
    Ymax = round(results.pandas().xyxy[0].ymax[i])

    # Class: Car, Bus, Truck
    if results.pandas().xyxy[0].name[i] == "car" or results.pandas().xyxy[0].name[i] == "truck" or results.pandas().xyxy[0].name[i] == "bus":
      for j in range(len(parking_Point)):
        # Parking Position about X,Y & Parking Flag is false
        if parking_Point[j][0] > Xmin and parking_Point[j][0] < Xmax and parking_Point[j][1] > Ymin and parking_Point[j][1] < Ymax and parking_Flag[j] == False:
            parking_Flag[j] = True
            car_Count += 1
            cv.rectangle(src, (Xmin, Ymin), (Xmax, Ymax), RED, 2)
            break
    else:
      continue

  # When it is available parking space, Draw parking area to green
  for i in range(len(parking_Flag)):
    if parking_Flag[i] == False:
      parking_Array = np.array([[parking_Coordinate[i][0]],[parking_Coordinate[i][1]],[parking_Coordinate[i][2]],[parking_Coordinate[i][3]]])
      mask = src.copy()
      cv.fillConvexPoly(src, parking_Array, GREEN)
      src = addWeighted(src, 1-TRANSPARENCY, mask, TRANSPARENCY, gamma=0)

  # Press Esc to Exit, Stop Video to 's' 
  k = cv.waitKey(delay) & 0xFF
  if k == 27:
    break
  elif k == ord('s'):
    cv.waitKey()
  
  # Count Text
  Count_Text = f"Number of Car: {car_Count}"
  putText(src, Count_Text, (100, 120), USER_FONT, 1, RED)

  # Write the Parking Count at Text File
  f.write(f"{' '*3}{frame_Num}\t{' '*7}{car_Count}\n")
  
  # Update
  frame_Num += 1

  # Time Loop End
  endTime = cv.getTickCount()

  # FPS Calculate
  FPS = round(getTickFrequency()/(endTime - startTime))

  # FPS Text
  FPS_Text = f"FPS: {FPS}"
  putText(src, FPS_Text, (100, 160), USER_FONT, 1, RED)

  # Show result image
  imshow("source", src)

  # Record Video
  out.write(src)

# Release
cv.destroyAllWindows()
cap.release()
out.release()
f.close()