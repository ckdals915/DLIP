"""## Import OpenCV Library"""

import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from cv2 import *
from cv2.cv2 import *


# Check open-cv version
print(cv.__version__)

class MouseGesture():
    def __init__(self) -> None:
        self.is_dragging = False
        self.x0, self.y0, self.w0, self.h0 = -1,-1,-1,-1
    
    def on_mouse(self, event, x, y, flags, param):
        if event == EVENT_LBUTTONDOWN:
            value = param[y, x]
            print(f"Press left button \t x: {x} y: {y} pixel: {value}")
        return

# Open the Video
# Press `esc` to stop video 
# cap = cv.VideoCapture("road_straight_student.mp4")
cap = cv.VideoCapture("road_lanechange_student.mp4")

# Variable
slope_left, slope_right = 1.0, -1.0
right_recognize_count, left_recognize_count = 0, 0
left_max = [0,0]
right_max = [0,0]
left_y0 = 0
right_y0 = 0

# Line & Circle Spec
white = (255, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)

line_width = 4
circle_radius = 5


while(1):
    # Check Window Time(FPS)

    # Read Video Capture
    cap_flag, src = cap.read()

    if cap_flag != 1:
        break
    # Vanishing Point
    rows, cols, _ = src.shape
    vanishing_point = [int(cols/2), int(rows/2)]

    # Canny Edge(Gaussian Filter)
    src_canny = Canny(src, 100, 250, None, 3)

    # ROI Setting
    
    roi = np.zeros_like(src_canny, np.uint8)
    roi_point = np.array([[559,430], [694, 430], [931, 620], [213, 620]])
    roi = fillConvexPoly(roi, roi_point, white)
    
    src_roi = np.zeros_like(src_canny, np.uint8)

    bitwise_and(src_canny, src_canny, src_roi, roi)
    
    # imshow('fillConvexPoly', src_roi)

    imshow("roi", src_roi)

    # Hough Transform
    cdstP = cvtColor(src_roi, COLOR_GRAY2BGR)
    linesP = HoughLinesP(src_roi, 1.0, np.pi/180, 50, _, 50, 100)
    
    # Initialization flag and slope counting
    left_count, right_count = 0, 0
    sum_slope_left, sum_slope_right, sum_left, sum_right = 0.0, 0.0, 0, 0
    right_slope_flag, left_slope_flag = False, False
    

    # Store left & right slope
    slope_left_temp = slope_left
    slope_right_temp = slope_right
    left_y0_temp = left_y0
    right_y0_temp = right_y0


    if linesP is not None:
        for i in range(len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
            imshow("LINEP", cdstP)
            # slope
            slope_temp = float((l[3] - l[1])/(l[2] - l[0]))
            if slope_temp <= 0.0:
                if left_max[1] <= l[1]:
                    left_max[0] = l[0]
                    left_max[1] = l[1]
                elif left_max[1] <= l[3]:
                    left_max[0] = l[2]
                    left_max[1] = l[3]
                sum_slope_left += slope_temp
                left_count += 1
                left_slope_flag = True

            elif slope_temp > 0.0:
                if right_max[1] <= l[1]:
                    right_max[0] = l[0]
                    right_max[1] = l[1]
                elif right_max[1] <= l[3]:
                    right_max[0] = l[2]
                    right_max[1] = l[3]
                sum_slope_right += slope_temp
                right_count += 1
                right_slope_flag = True

    
    # **************** Calculate each line, fitting line ********************
    # Left, Right line
    if left_slope_flag == True:
        slope_left = float(sum_slope_left / left_count)
        left_y0 = int(float(left_max[1]) - slope_left * float(left_max[0]))
        left_recognize_count = 0

    elif left_slope_flag == False:
        slope_left = slope_left_temp
        left_y0 = left_y0_temp
        left_recognize_count += 1

    if right_slope_flag == True:
        slope_right = sum_slope_right / float(right_count)
        right_y0 = int(float(right_max[1]) - slope_right * float(right_max[0]))
        right_recognize_count = 0

    elif right_slope_flag == False:
        slope_left = slope_left_temp
        right_y0 = right_y0_temp
        right_recognize_count += 1
    
    # Vanishing Point
    vanishing_point[0] = int(-1.0 * (float(left_y0 - right_y0)) / (slope_left - slope_right))
    vanishing_point[1] = int(-1.0 * slope_left * (float(left_y0 - right_y0)) / (slope_left - slope_right)  + float(left_y0))

    print(f"slope_left: {slope_left}, slope_right: {slope_right}")
    # print(f"left y0: {left_y0}, right y0: {right_y0}")
    # print(f"left count: {left_count}, right count: {right_count}")
    # print(f"left recognize: {left_recognize_count}, right recognize: {right_recognize_count}")

    # Make a fitted line
    cv.line(src, (left_max[0], left_max[1]), (vanishing_point[0], vanishing_point[1]), (0, 0, 255), line_width, cv.LINE_AA)
    cv.line(src, (right_max[0], right_max[1]), (vanishing_point[0], vanishing_point[1]), (0, 0, 255), line_width, cv.LINE_AA)
    cv.circle(src, (vanishing_point[0], vanishing_point[1]), circle_radius, green, -1, FILLED)

    imshow("HoughP", cdstP)
    imshow("source", src)
    mouse_class = MouseGesture()
    setMouseCallback('source', mouse_class.on_mouse, param=src)
    

    # If Line is NOT detected, Save previous lines

    # Predict direction of car

    # Draw line and filled in polygon
    
    
    # cv.imshow('src',src)


    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break


cv.destroyAllWindows()
cap.release()

