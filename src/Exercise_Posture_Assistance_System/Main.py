'''
* *****************************************************************************
* @author   ChangMin An, JiWoo Yang
* @Mod      2022 - 06 - 17
* @brief   DLIP LAB: Gym Accident Prevention System 
* *****************************************************************************
'''
#===============================================#
#              Open Library Declare             #
#===============================================#
import tensorflow as tf
import numpy as np
import cv2 as cv
from cv2 import *
from cv2.cv2 import *
from tkinter import *

#===============================================#
#                Global Variable                #
#===============================================#

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
USER_FONT       = FONT_HERSHEY_DUPLEX
fontScale       = 1
fontThickness   = 2

# Definition Body Parts
LEFT_SHOULDER       = 5
RIGHT_SHOULDER      = 6
LEFT_ELBOW          = 7
RIGHT_ELBOW         = 8
LEFT_WRIST          = 9
RIGHT_WRIST         = 10

# Definition of body edges
EDGES = {
    (LEFT_SHOULDER,LEFT_ELBOW): 'm',
    (LEFT_ELBOW,LEFT_WRIST): 'm',
    (RIGHT_SHOULDER,RIGHT_ELBOW): 'c',
    (RIGHT_ELBOW,RIGHT_WRIST): 'c',
    (LEFT_SHOULDER,RIGHT_SHOULDER): 'y',  
}

# Thresholding
CONFIDENCE_THRESHOLD    = 0.2   # For minimum confidence of output
CORRECT_RECOGNIGION     = 5     # 

START_THRESHOLD         = 0.3   # For value of good pose
START_COUNT_THRESH      = 30    # For start counting

# User input
inputCount      = 11 # For user input

# Flag
system_Flag     = False                        # For start system
start_Flag      = False                        # For start counting process
finish_Flag     = False                        # For finish workout
tk_Flag         = False                        # For new user input
up_down_Flag    = [False, False, False, False] # 0: down_Flag, 1: up_Flag, 2: left_down_Flag, 3: right_down_Flag

# For counting
counting_List   = [0, 0, 0, 0, 0.0, 0.0, 0] # 0: Good count, 1: bad count, 2: left_E_ystack, 3: right_E_ystack, 4: left_E_avg, 5: right_E_avg 6: count_frame
start_Count     = 0                         # For starting setting

# 0:set Count, 1: startCount, 2: userSet
offset_Text = ""

# For Balance
balance_List = [0.0, 10.0, 0.0] # 0: balance, 1: good, 2: bad


# Frame count
frame_Num           = 0         # for processing
position_FrameList  = [0, 0]    # 0: worst / 1: Best

# Video naming
Video = "DEMO.mp4"

#===============================================#
#              Definition Function              #
#===============================================#
# Get shape of keypoint and buffer of thing that we need(shoulder, elbow, wrist position) 
def Get_Shape_PoseBuf(frame, keypoints, edge, confidence_threshold):
   # Rendering
    y, x, c = frame.shape
    _shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    _poseBuf = []
    for edge, color in EDGES.items():
        
        p1, p2 = edge
        y1, x1, c1 = _shaped[p1]
        y2, x2, c2 = _shaped[p2]

        if(c1 > confidence_threshold) & (c2 > confidence_threshold):
            # Buffer Pose Coordinate
            _poseBuf.append([[p1, int(x1), int(y1)],[p2, int(x2), int(y2)]])

    # Return
    return _poseBuf, _shaped

# Draw skeletone model
def Draw_Connecting(frame, _shaped, edge, confidence_threshold):
    for edge, color in EDGES.items():
        
        p1, p2 = edge
        y1, x1, c1 = _shaped[p1]
        y2, x2, c2 = _shaped[p2]

        if(c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),BLUE, 2)
            cv.circle(frame, (int(x1), int(y1)), 4, GREEN, -1)
            cv.circle(frame, (int(x2), int(y2)), 4, GREEN, -1)

# Get Start Flag and start_count for counting, offset_Text and gage bar
def Start_Postion_Adjustment(_frame, _poseBuf, _startCount, _startFlag,  _offsetText):
    if _startCount >= START_COUNT_THRESH:
        _startFlag      = True
    
    if _startFlag == False:
        if len(_poseBuf) == CORRECT_RECOGNIGION:
            # print(pose_Buf)
            for i in range(len(_poseBuf)):
                if _poseBuf[i][0][0] == LEFT_SHOULDER:
                    x1_left = _poseBuf[i][0][1]
                    y1_left = _poseBuf[i][0][2]
                elif _poseBuf[i][1][0] == LEFT_WRIST:
                    x2_left = _poseBuf[i][1][1]
                    y2_left = _poseBuf[i][1][2]
                elif _poseBuf[i][0][0] == RIGHT_SHOULDER:
                    x1_right = _poseBuf[i][0][1]
                    y1_right = _poseBuf[i][0][2]
                elif _poseBuf[i][1][0] == RIGHT_WRIST:
                    x2_right = _poseBuf[i][1][1]
                    y2_right = _poseBuf[i][1][2]
            slope_left      = abs(round(float((y2_left-y1_left)/(x2_left-x1_left+0.000001)), 3))
            slope_right     = abs(round(float((y2_right-y1_right)/(x2_right-x1_right+0.000001)), 3))
            slope_offset    = abs(slope_left - slope_right)
        
            # Counting
            if y1_left / y2_left > 1.5 and y1_right / y2_right > 1.5:
                # _skeletoneFlag = True
                if slope_offset < START_THRESHOLD:
                    _startCount += 1
                    _offsetText = "Correct Position! Stop it"
                elif slope_offset >= START_THRESHOLD:
                    if slope_right > slope_left:
                        _offsetText = "Move your hands to the RIGHT"
                    elif slope_right < slope_left:
                        _offsetText = "Move your hands to the LEFT"
                    _startCount = 0
                if _startCount !=0:
                    cv.rectangle(_frame, (100, 60), (100+13*_startCount, 90), GREEN, -1)
                    cv.rectangle(_frame, (100, 60), (100+13*30, 90), BLACK, 3)

    return _startFlag, _startCount, _offsetText

# Calculate the balance value
def Calculate_Balance(_shaped, _balance):
    Left_w_y    = 0.0
    Right_w_y   = 0.0
    Left_w_y, _, Left_w_c       = _shaped[9]  # Left wrist
    Right_w_y, _, Right_w_c     = _shaped[10] # Right wrist
    # Valance > 0 : left wrist under the right wrist
    # Valance < 0 : right wrist under the left wrist
    if(Left_w_y != 0) & (Right_w_y != 0) & (Left_w_c > CONFIDENCE_THRESHOLD) & (Right_w_c > CONFIDENCE_THRESHOLD):
        _balance = Left_w_y/Right_w_y - 1.0 
    return _balance

# Count the workout number
def Count_Workout(_frame, _shaped, _inputCount, _finishFlag, _countList, _balanceList, _updownflag):
    _countList[6] += 1
    
    Left_E_y, _, _          = _shaped[7]    # Left Elbow
    Right_E_y, _, _         = _shaped[8]    # Right Elbow

    _countList[2]    = _countList[2] + Left_E_y # left-Stack
    _countList[3]    = _countList[3] + Right_E_y # right-Stack


    if _countList[6]%10 == 0:
        _countList[6]       = 0
        _countList[4]       = _countList[2]/10.0      # Left_E_avg
        _countList[5]       = _countList[3]/10.0      # Right_E_avg
        _countList[2]       = 0                       # left-Stack initialize
        _countList[3]       = 0                       # right-Stack initialize
        # 0: down_Flag, 1: up_Flag, 2: left_down_Flag, 3: right_down_Flag, 4:left_up_Flag, 5: right_up_Flag
        # up/down flag control
        if _updownflag[0] == False: # 일단 내려갔다 라는 것에 대한 조건
            if _updownflag[2] == False: # 왼쪽이 내려갔는지    
                if _countList[4]>shaped[LEFT_SHOULDER][0]:
                    _updownflag[2] = True
                    # Find Worst pose
                    if abs(_balanceList[0]) > abs(_balanceList[2]):
                        _balanceList[2] = _balanceList[0]
                        imwrite('WorstPose.jpg', _frame)
            if _updownflag[3] == False: # 오른쪽이 내려갔는지    
                if _countList[5]>shaped[RIGHT_SHOULDER][0]:
                    _updownflag[3] = True
                    # Find Worst pose
                    if abs(_balanceList[0]) > abs(_balanceList[2]):
                        _balanceList[2] = _balanceList[0]
                        imwrite('WorstPose.jpg', _frame)

            if _updownflag[2] == True and _updownflag[3] == True: # 둘다 내려갔는지
                _updownflag[0] = True
                # Find Worst pose
                if abs(_balanceList[0]) < abs(_balanceList[1]):
                    _balanceList[1] = _balanceList[0]
                    imwrite('BestPose.jpg', _frame)
                print("Two hand Down")

        if _updownflag[0] == False:
            if _updownflag[3] == True:
                if _countList[5]<shaped[RIGHT_SHOULDER][0]:
                    _countList[1]+=1
                    _updownflag[3] = False
                    _updownflag[2] = False
            if _updownflag[2] == True:
                if _countList[4]<shaped[LEFT_SHOULDER][0]:
                    _countList[1]+=1
                    _updownflag[2] = False
                    _updownflag[3] = False
        else:
            if _countList[5]<shaped[RIGHT_SHOULDER][0] and _countList[4]<shaped[LEFT_SHOULDER][0]:
                if _balanceList[0] < -0.2 or _balanceList[0] > 0.2:
                    _countList[1]+=1
                else:
                    _countList[0]+=1
                _updownflag[0] = False
                _updownflag[2] = False
                _updownflag[3] = False

    # For finish Flage
    if _countList[0] == _inputCount:
        _finishFlag = True
    
    return _finishFlag, _countList, _balanceList, _updownflag, _countList[6]

# Reset all parameter
def ResetPram():
    # System flag
    global system_Flag, start_Flag, finish_Flag, up_down_Flag, counting_List, start_Count, offset_Text, balance_List, frame_Num, inputCount, frame_Num2, position_FrameList

    # System flag
    system_Flag     = False
    start_Flag      = False
    finish_Flag     = False

    # For workout count
    # 0: down_Flag, 1: up_Flag, 2: left_down_Flag, 3: right_down_Flag, 4:left_up_Flag, 5: right_up_Flag
    up_down_Flag    = [False, False, False, False]

    # 0: Good count, 1: bad count, 2: left_E_ystack, 3: right_E_ystack, 4: left_E_avg, 5: right_E_avg 6: count_frame
    counting_List   = [0, 0, 0, 0, 0.0, 0.0, 0]

    # For starting setting
    # 0:set Count, 1: startCount, 2: userSet
    start_Count = 0
    offset_Text = ""

    # Balance_List
    # 0: balance, 1: good, 2: bad)
    balance_List = [0.0, 10.0, 0.0]

    # Frame count
    position_FrameList  = [0, 0]    # 0: worst / 1: Best
#===============================================#
#                   Show Text                   #
#===============================================#

def show_Text(_img, _Balance, _count, _countBad, _flag):
    if _flag == False:
        # All workout count
        TEXT_allCount          = f"All count = {_count+_countBad}"
        # Good workout count
        TEXT_goodCount          = f"Good count = {_count}/{inputCount}"
        # Bad workout count
        TEXT_bedCount       = f"Bad count = {_countBad}"
        # Balance value
        TEXT_Bal_value      = f"Balance = {round(_Balance,3)}"
        # Balance Feedback
        if _Balance <-0.2:
            TEXT_Bal_Fed    = f"FeedBack: Push down left side"
            color = RED
        elif _Balance >0.2:
            TEXT_Bal_Fed    = f"FeedBack: Push down right side"
            color = RED
        else:
            TEXT_Bal_Fed    = f"FeedBack: Good pos bro!!."
            color = WHITE

        textSize, _ = cv.getTextSize(TEXT_Bal_Fed, USER_FONT, fontScale, fontThickness)

        # Print Text
        cv.rectangle(_img, (_img.shape[0]-530, 0), (_img.shape[0], 200), BLACK, -1)
        cv.putText(_img, TEXT_allCount, (_img.shape[0]-530, textSize[1]), USER_FONT, fontScale, WHITE, fontThickness)
        cv.putText(_img, TEXT_goodCount, (_img.shape[0]-530, textSize[1]+40), USER_FONT, fontScale, WHITE, fontThickness)
        cv.putText(_img, TEXT_bedCount, (_img.shape[0]-530, textSize[1]+80), USER_FONT, fontScale, WHITE, fontThickness)
        cv.putText(_img, TEXT_Bal_value, (_img.shape[0]-530, textSize[1]+120), USER_FONT, fontScale, color, fontThickness)
        cv.putText(_img, TEXT_Bal_Fed, (_img.shape[0]-530, textSize[1]+160), USER_FONT, fontScale, color, fontThickness)
    else:
        TEXT_finish = f"Finish!! Good Job brother~~."
        
        textSize, _ = cv.getTextSize(TEXT_finish, USER_FONT, fontScale, fontThickness)
        cv.rectangle(_img, (int(_img.shape[0]/2.0)-int(textSize[0]/2.0)-5, int(_img.shape[1]/2)-textSize[1]-5), (int(_img.shape[0]/2.0)+int(textSize[0]/2.0)+5, int(_img.shape[1]/2)+5), BLACK, -1)
        cv.putText(_img, TEXT_finish, (int(_img.shape[0]/2.0)-int(textSize[0]/2.0), int(_img.shape[1]/2)), USER_FONT, fontScale, RED, fontThickness)

def show_Start_text(_img, _text):
    cv.putText(_img, _text, (100, 50), USER_FONT, fontScale, WHITE, fontThickness) 
    
#===============================================#
#                      Main                     #
#===============================================#

# Model Interpreter Definition
interpreter = tf.lite.Interpreter(model_path = 'lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# TKinter for USER Count
tk = Tk()
tk.title('Input Exercise Count')
tk.geometry("200x100")

def Input_Count():
    global set_List
    global inputCount
    A = int(entry1.get())
    inputCount = A

label1 = Label(tk, text='Input Count').grid(row=0, column=0)
entry1 = Entry(tk)
entry1.grid(row=0,column=1)

btn = Button(tk, text='Press Count', bg='black', fg='white', command=Input_Count).grid(row=1,column=0)
exit_button = Button(tk, text='Exit', bg='black', fg='white', command=tk.destroy).grid(row=1,column=1)

tk.mainloop()

# Open the Video & Recording Video Configuration
# cv.VideoCaputure(0) -> notebook cam
# cv.VideoCaputure(1) -> another cam connecting with my notebook
# cv.VideoCapture(filename.mp4) -> Video
cap = cv.VideoCapture("DEMO.mp4")

#
w = round(cap.get(CAP_PROP_FRAME_WIDTH))
h = round(cap.get(CAP_PROP_FRAME_HEIGHT))
fps = cap.get(CAP_PROP_FPS)
fourcc = VideoWriter_fourcc(*'DIVX')
out = VideoWriter('output.avi', fourcc, fps, (w,h))
delay = round(1000/fps)

if (cap.isOpened() == False): # if there is no video we can open, print error
  print("Not Open the VIDEO")

#================== While Loop =================#
while cap.isOpened():
    # When you press the 'r' botton -> restart
    if tk_Flag == True:
        tk = Tk()
        tk.title('Input Exercise Count')
        tk.geometry("200x100")

        label1 = Label(tk, text='Input Count').grid(row=0, column=0)
        entry1 = Entry(tk)
        entry1.grid(row=0,column=1)

        btn = Button(tk, text='Press Count', bg='black', fg='white', command=Input_Count).grid(row=1,column=0)
        exit_button = Button(tk, text='Exit', bg='black', fg='white', command=tk.destroy).grid(row=1,column=1)
        tk.mainloop()

        ResetPram()
        tk_Flag = False

    frame_Num += 1

    # Start Window Time
    startTime = cv.getTickCount()

    # Video Read
    ret, frame = cap.read()
    if ret == False:
        print("Video End")
        break

    # Reshape image
    frame       = resize(frame, dsize = (1080, 1080), interpolation=INTER_LINEAR)
    img         = frame.copy()
    img         = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    
    #Setup input and Output
    input_details   = interpreter.get_input_details()   # receive information of input tensor
    output_details  = interpreter.get_output_details()   # receive information of output tensor
    
    # input to model
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    # Get output to model
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # =================== START POINT =================== #
    pose_Buf = []
    pose_Buf, shaped = Get_Shape_PoseBuf(frame, keypoints_with_scores, EDGES, CONFIDENCE_THRESHOLD)

    if shaped[RIGHT_WRIST][0]>shaped[RIGHT_ELBOW][0] and shaped[LEFT_WRIST][0]>shaped[LEFT_ELBOW][0]: # 쉴 때의 조건
        system_Flag = False
    else:
        system_Flag = True

    if system_Flag == True:
        # Get Start Flag and start_count for counting, offset_Text
        if finish_Flag == False:
            start_Flag, start_Count, offset_Text = Start_Postion_Adjustment(frame, pose_Buf, start_Count, start_Flag, offset_Text)
            if start_Flag == False:
                show_Start_text(frame, offset_Text)

        # draw skeleton
        Draw_Connecting(frame, shaped, EDGES, CONFIDENCE_THRESHOLD)
            
        # Start count processing
        if start_Flag == True and finish_Flag == False:
            balance_List[0] = Calculate_Balance(shaped, balance_List[0])
            finish_Flag, counting_List, balance_List, up_down_Flag, counting_List[6] = Count_Workout(frame, shaped, inputCount, finish_Flag, counting_List, balance_List, up_down_Flag)
            
            show_Text(frame, balance_List[0], counting_List[0], counting_List[1], finish_Flag)
        # Finish Flag
        elif finish_Flag == True:
            show_Text(frame, balance_List[0], counting_List[0], counting_List[1], finish_Flag)
            system_Flag = False
        
        # Press Esc to Exit, Stop Video to 's' 
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
        elif k == ord('s'):
            cv.waitKey()
        elif k == ord('r'):
            tk_Flag = True

        # Time Loop End
        endTime = cv.getTickCount()

        # FPS Calculate
        FPS = round(getTickFrequency()/(endTime - startTime))
    
        # FPS Text
        FPS_Text = f"FPS: {FPS}"
        putText(frame, FPS_Text, (0, 20), USER_FONT, 0.8, RED)    

        cv.imshow('MoveNet Lightning', frame)
        resizeWindow('MoveNet Lightning', 1080, 1080)
    else:
        if finish_Flag == False:
            ResetPram()
        else:
            break
        # Press Esc to Exit, Stop Video to 's' 
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
        elif k == ord('s'):
            cv.waitKey()
        elif k == ord('r'):
            tk_Flag = True

        # Time Loop End
        endTime = cv.getTickCount()

        # FPS Calculate
        FPS = round(getTickFrequency()/(endTime - startTime))
        
        # FPS Text
        FPS_Text = f"FPS: {FPS}"
        putText(frame, FPS_Text, (0, 20), USER_FONT, 0.8, RED)    

        cv.imshow('MoveNet Lightning', frame)

    # # Record Video
    out.write(frame)

cap.release()
cv.destroyAllWindows()
out.release()

# ============================================================================== #
# =============================== Final result report ========================== #
# ============================================================================== #
if finish_Flag == True:
    # image stack
    best_image      = cv.imread('BestPose.jpg')
    best_image      = cv.resize(best_image, (0,0), None, .5, .5)
    worst_image     = cv.imread('WorstPose.jpg')
    worst_image     = cv.resize(worst_image, (0,0), None, .5, .5)
    pose_result     = np.vstack((best_image, worst_image))
    result_paper    = np.zeros_like(pose_result)
    workout_result  = np.hstack((result_paper, pose_result))

    # Put text
    # Text for result image
    TEXT_GOOD       = f"Best Pose(Balance {abs(round(balance_List[1],3))})"
    TEXT_BED        = f"Worst Pose(Balance {abs(round(balance_List[2],3))})"

    # Text for result report
    TEXT_RESULT1         = f"======================"
    TEXT_RESULT          = f"---------Result---------"
    TEXT_RESULT2         = f"======================"
    TEXT_GOOD_COUNT     = f"Count about Good Pose = {counting_List[0]}"
    TEXT_BED_COUNT      = f"Count about Bad Pose = {counting_List[1]}"
    TEXT_COUNT          = f"Count about All Pose = {counting_List[0]+counting_List[1]}"

    TEXT_RATIO          = f"Performace ratio ="

    # Parameter for position
    Size_GOOD, _    = cv.getTextSize(TEXT_GOOD, USER_FONT, fontScale, fontThickness)
    Size_BED, _     = cv.getTextSize(TEXT_BED, USER_FONT, fontScale, fontThickness)
    width           = best_image.shape[0] # best image width
    height          = best_image.shape[1] # best image height
    x_GOOD          = Size_GOOD[0]
    y_GOOD          = Size_GOOD[1]
    x_BED           = Size_BED[0]
    y_BED           = Size_BED[1]

    # Draw ract and Put Text for image
    cv.rectangle(workout_result, (width, 0), (width*2, y_GOOD+13), WHITE, -1)
    cv.rectangle(workout_result, (width, height), (width*2, y_GOOD+height+13), WHITE, -1)
    cv.putText(workout_result, TEXT_GOOD, (width+int(width/2)-int(x_GOOD/2), y_GOOD+5), USER_FONT, fontScale, BLACK, fontThickness)
    cv.putText(workout_result, TEXT_BED, (width+int(width/2)-int(x_BED/2), y_GOOD+height+5), USER_FONT, fontScale, BLACK, fontThickness)
    # Put Text for result report
    cv.putText(workout_result, TEXT_RESULT1, (0, y_GOOD), USER_FONT, fontScale, RED, fontThickness)
    cv.putText(workout_result, TEXT_RESULT, (0, y_GOOD*2), USER_FONT, fontScale, RED, fontThickness)
    cv.putText(workout_result, TEXT_RESULT2, (0, y_GOOD*3), USER_FONT, fontScale, RED, fontThickness)

    cv.putText(workout_result, TEXT_COUNT, (0, y_GOOD*4), USER_FONT, fontScale, WHITE, fontThickness)
    cv.putText(workout_result, TEXT_GOOD_COUNT, (0, y_GOOD*6), USER_FONT, fontScale, WHITE, fontThickness)
    cv.putText(workout_result, TEXT_BED_COUNT, (0, y_GOOD*8), USER_FONT, fontScale, WHITE, fontThickness)
    

    cv.imshow('Final result of your workout', workout_result)
    waitKey()
    cv.imwrite("Finish.jpg", workout_result)

    cv.destroyAllWindows()