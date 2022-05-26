'''
* *****************************************************************************
* @author	ChangMin An
* @Mod		2022 - 05 - 01
* @brief	DLIP LAB: Straight Lane Detection and Departure Warning
* *****************************************************************************
'''

#===============================================#
#              Open Library Declare             #
#===============================================#
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from cv2 import *
from cv2.cv2 import *
import copy

#===============================================#
#                Global Variable                #
#===============================================#
# Video = "road_straight_student.mp4"
Video = "road_lanechange_student.mp4"
# Video = "road_validation_1sec.mp4"

# Color Definition (BGR)
WHITE               = (255, 255, 255)
RED                 = (  0,   0, 255)
GREEN               = (  0, 255,   0)
PINK                = (184,  53, 255)
YELLOW              = (  0, 255, 255)
BLUE                = (255,   0,   0)
BLACK               = (  0,   0,   0)
PURPLE              = (255, 102, 102)

# Line Definition
DEG2RAD             = np.pi/180
RAD2DEG             = 180/np.pi
ANGLE_LEFT          = -30
ANGLE_RIGHT         = 30
LINE_WIDTH          = 4
CIRCLE_RADIUS       = 5
BIAS_LENGTH         = 20
TRANSPARENCY        = 0.3

WARNING_COUNT       = 20
SLOPE_OFFSET        = 0.0001
LANE_WARNING        = 20

# Font Definition
USER_FONT           = FONT_HERSHEY_DUPLEX

# Length Definition
ARROW_LENGTH        = 100
RECOGNITION_LENGTH  = 100

# Hough Transform
houghThresh          = 50

# Line Processing
left_min                 = 0.0
right_Max                = 0.0
left_Coordinate          = [300, 720, 640, 360]
right_Coordinate         = [900, 720, 640, 360]
left_Coordinate_Temp     = [300, 720, 640, 360]
right_Coordinate_Temp    = [900, 720, 640, 360]
vanishing_Point          = [640, 360]
vanishing_Point_Temp     = [640, 360]

# Line Detection
left_Count          = 0
right_Count         = 0
bias_Direction      = 'RIGHT'
bias_Text           = "BIAS: -"

# Time Count
startTime           = 0
endTime             = 0

# Font
font_Color          = BLACK
status_Color        = GREEN

#===============================================#
#                     Main                      #
#===============================================#

# Open the Video
cap = cv.VideoCapture(Video)
if (cap.isOpened() == False):
    print("NOT Open the VIDEO")

#================== While Loop =================#                  
while(1):
    # Start Window Time
    startTime = cv.getTickCount()

    # Read Video Capture
    cap_Flag, src = cap.read()
    
    # If Capture is failed, break the loop
    if cap_Flag == False:
        print("Video End")
        break
    
    # Pre-Processing
    src_filtered = GaussianBlur(src, (5,5), 0)
    
    # Canny Edge Detection
    src_canny = Canny(src_filtered, 50, 250, None, 3)
    
    # ROI Setting
    roi = np.zeros_like(src_canny, np.uint8)
    src_roi = np.zeros_like(src_canny, np.uint8)
    roi_point = np.array([[560,430], [695, 430], [1100, 620], [210, 620]])
    roi = fillConvexPoly(roi, roi_point, WHITE)
    bitwise_and(src_canny, src_canny, src_roi, roi)
    
    # Hough Transform (HoughLinesP)
    linesP = HoughLinesP(src_roi, 1, DEG2RAD, houghThresh, 1, 10, 1000)  

    #================== Line Processing =================#

    # Initialization Variable
    slope, left_min, right_Max = 0.0, 0.0, 0.0
    left_Coordinate_Temp = copy.deepcopy(left_Coordinate)
    right_Coordinate_Temp = copy.deepcopy(right_Coordinate)
    vanishing_Point_Temp = copy.deepcopy(vanishing_Point)

    # Initialization
    left_Flag = False
    right_Flag = False
    circle_Color, area_Color, text_Color = GREEN, GREEN, BLACK        # Color Definition

    if linesP is not None:
        for i in range(len(linesP)):
            l = linesP[i][0]
            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]

            # Slope
            slope = float((y2 - y1) / ((x2 - x1) + SLOPE_OFFSET))
            
            # Condition of Slope range(LEFT)
            if slope > np.tan(-85 * DEG2RAD) and slope < np.tan(ANGLE_LEFT * DEG2RAD):
                left_Flag = True
                if slope < left_min:
                    left_min = slope
                    left_Coordinate = [x1, y1, x2, y2]
            
            # Condition of Slope range(RIGHT)
            elif slope < np.tan(85 * DEG2RAD) and slope > np.tan(ANGLE_RIGHT * DEG2RAD):
                right_Flag = True
                if slope > right_Max:
                    right_Max = slope
                    right_Coordinate = [x1, y1, x2, y2]

    # Equation of the Straight Line(LEFT)
    if left_Flag == True:
        m1 = float((left_Coordinate[1] - left_Coordinate[3]) / (left_Coordinate[0] - left_Coordinate[2]))
        b1 = float(left_Coordinate[1] - m1 * left_Coordinate[0])
        left_Color = BLUE            # Color Recognization
        left_Count = 0               # Initialization Recognize Counting

    elif left_Flag == False:
        m1 = float((left_Coordinate_Temp[1] - left_Coordinate_Temp[3]) / (left_Coordinate_Temp[0] - left_Coordinate_Temp[2]))
        b1 = float(left_Coordinate_Temp[1] - m1 * left_Coordinate_Temp[0])
        left_Color = YELLOW          # Color Recognization
        left_Count += 1              # Recognize Counting


    # Equation of the Straight Line(RIGHT)
    if right_Flag == True:
        m2 = float((right_Coordinate[1] - right_Coordinate[3]) / (right_Coordinate[0] - right_Coordinate[2]))
        b2 = float(right_Coordinate[1] - m2 * right_Coordinate[0])
        right_Color = BLUE           # Color Recognization
        right_Count = 0              # Recognize Counting

    elif right_Flag == False:
        m2 = float((right_Coordinate_Temp[1] - right_Coordinate_Temp[3]) / (right_Coordinate_Temp[0] - right_Coordinate_Temp[2]))
        b2 = float(right_Coordinate_Temp[1] - m2 * right_Coordinate_Temp[0])
        right_Color = YELLOW          # Color Recognization
        right_Count += 1              # Recognize Counting

    # Line Starting Point
    Ymax, Xmax, _ = src.shape
    left_Start = [int((Ymax - b1)/m1), Ymax]
    right_Start = [int((Ymax - b2)/m2), Ymax]
    lane_Length = abs(left_Start[0] - right_Start[0])

    # Vanishing Point
    if left_Count > WARNING_COUNT or right_Count > WARNING_COUNT or lane_Length < RECOGNITION_LENGTH:
        vanishing_Point[0] = vanishing_Point_Temp[0]
        vanishing_Point[1] = vanishing_Point_Temp[1]
        left_Color, right_Color, circle_Color, area_Color = RED, RED, RED, RED

    else:
        vanishing_Point = [int((b2-b1)/(m1-m2)), int((float(b2-b1))*m1/(m1-m2)+float(b1))]

    # Draw the Direction Area
    direction_Array = np.array([vanishing_Point, left_Start, right_Start])    
    mask = src.copy()
    cv.fillConvexPoly(mask, direction_Array, area_Color)
    src = addWeighted(src, 1-TRANSPARENCY, mask, TRANSPARENCY, gamma=0)

    #========================== BIAS =======================#

    # Bias between Mid-Point and Vanish-Point
    mid_Point = [int(Xmax/2), int(Ymax)]
    vehicle_Point = int((left_Start[0] + right_Start[0])/2)
    bias = round((vehicle_Point - mid_Point[0]) / mid_Point[0] * 100.0, 2)    # [%]

    # Draw the Bias Line
    if left_Count <= WARNING_COUNT and right_Count <= WARNING_COUNT and lane_Length > RECOGNITION_LENGTH:
        cv.line(src, (mid_Point[0], mid_Point[1]),(mid_Point[0], mid_Point[1] - BIAS_LENGTH), WHITE, LINE_WIDTH, cv.LINE_AA)
        cv.line(src, (vehicle_Point, mid_Point[1]),(vehicle_Point, mid_Point[1] - BIAS_LENGTH), PINK, LINE_WIDTH, cv.LINE_AA)

    # Draw the line & Vanishing Point
    if left_Count < WARNING_COUNT:
        cv.line(src, (left_Start[0], left_Start[1]),(vanishing_Point[0], vanishing_Point[1]), left_Color, LINE_WIDTH, cv.LINE_AA)
    if right_Count < WARNING_COUNT:
        cv.line(src, (right_Start[0], right_Start[1]),(vanishing_Point[0], vanishing_Point[1]), right_Color, LINE_WIDTH, cv.LINE_AA)
    cv.circle(src, (vanishing_Point[0], vanishing_Point[1]), CIRCLE_RADIUS, circle_Color, -1, FILLED)

    #========================== TEXT =======================#

    # BIAS Text    
    if left_Count > WARNING_COUNT or right_Count > WARNING_COUNT:
        font_Color = RED
        bias_Text = "BIAS: -"
    elif lane_Length > RECOGNITION_LENGTH:
        font_Color = BLACK 
        if bias >= 0.0:
            bias_Direction = "LEFT"
        else:
            bias_Direction = "RIGHT"     
        bias_Text = f"BIAS: {bias_Direction} {abs(bias)}%"
    else:
        font_Color = font_Color
        bias_Direction = bias_Direction
    
    # Print Bias
    putText(src, bias_Text, (100, 100), USER_FONT, 1, font_Color)

    # Lane Status
    if left_Count > WARNING_COUNT or right_Count > WARNING_COUNT: 
        status_Color = RED
        lane_Status = "LINE CHANGE"
    elif abs(bias) > LANE_WARNING:
        status_Color = PURPLE
        lane_Status = "WARNING"
    elif left_Count < WARNING_COUNT and right_Count < WARNING_COUNT and lane_Length > RECOGNITION_LENGTH:
        status_Color = GREEN
        lane_Status = "SAFE"
    else:
        status_Color = status_Color
        lane_Status = lane_Status
    
    lane_Text = f"Lane Status: {lane_Status}"
    
    # Print Lane Status
    putText(src, lane_Text, (100, 140), USER_FONT, 1, status_Color)

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
    putText(src, FPS_Text, (100, 180), USER_FONT, 1, BLACK)

    # Imshow the source
    imshow("source", src)

# Release
cv.destroyAllWindows()
cap.release()