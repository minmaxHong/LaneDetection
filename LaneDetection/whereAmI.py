import cv2
import numpy as np
import time

VIDEO_PATH = r"C:\Users\User\Desktop\LaneDetection\LaneDetection\no_gps_obstacle#13.avi"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("="*20)
    print("Camera Open Failed")
    print("="*20)

ret, frame = cap.read()

output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(f'output_GPS13#_NoObstacle-{time.time()}.mp4', fourcc, fps, (output_width, output_height))

position = (640, 320)  
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)  
line_type = 3

x1, y1, width1, height1 = int(frame.shape[1] * 0.20), int(frame.shape[0] * 0.8), 300, 200
x2, y2, width2, height2 = int(frame.shape[1] * 0.55), int(frame.shape[0] * 0.8), 300, 200  
thresh_center_x3, thresh_center_y3, width3, height3 = int(frame.shape[1] * 0.15 + 400), int(frame.shape[0] - 100), 50, 100
crosswalk_center_x4, crosswalk_center_y4, width4, height4 = 0, int(frame.shape[0] * 0.7), int(frame.shape[1]), int(frame.shape[0] - frame.shape[0] * 0.7)

lower_white = np.array([0, 0, 200], dtype=np.uint8)
upper_white = np.array([180, 30, 255], dtype=np.uint8)

lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

lane_info = "LANE 3" 
temp_lane_info = ""
start_time = time.time()
flag = False


def find_contour(roi_img, min_area = 1500):
    gray = cv2.cvtColor(roi_img.copy(), cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=5)
    thresh = cv2.threshold(blur_frame, 200, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crosswalk_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4 and min_area < area: 
            crosswalk_contours.append(contour)

    return crosswalk_contours       

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    roi1 = frame[y1:y1+height1, x1:x1+width1]
    roi2 = frame[y2:y2+height2, x2:x2+width2]
    roi3 = frame[thresh_center_y3: thresh_center_y3 + height3, thresh_center_x3: thresh_center_x3 + width3]
    roi4 = frame[crosswalk_center_y4: crosswalk_center_y4 + height4, crosswalk_center_x4: crosswalk_center_x4 + width4]
    
    left_white_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV), lower_white, upper_white))
    left_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow))

    right_white_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV), lower_white, upper_white))
    right_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow))
    
    # cv2.rectangle(frame, (x1, y1), (x1+width1, y1+height1), (0, 255, 0), 2)
    # cv2.rectangle(frame, (x2, y2), (x2+width2, y2+height2), (0, 255, 0), 2)
    # cv2.rectangle(frame, (thresh_center_x3, thresh_center_y3), (thresh_center_x3+width3, thresh_center_y3+height3), (0, 255, 0), 2)
    # cv2.rectangle(frame, (crosswalk_center_x4, crosswalk_center_y4), (crosswalk_center_x4+width4, crosswalk_center_y4+height4), (0, 255, 0), 2)  

    crossing_centerline_thres = 200
    crosswalk_exist = find_contour(roi4)
            
    if flag == False:     
        if len(crosswalk_exist) > 0 and len(crosswalk_exist) < 10:
            if lane_info == "LANE 3":
                if right_white_pixel > right_yellow_pixel:
                    lane_info = "LANE 2"

            elif lane_info == "LANE 2":
                if left_white_pixel < left_yellow_pixel:
                    lane_info = "LANE 1"
                elif right_white_pixel < right_yellow_pixel:
                    lane_info = "LANE 3"

            elif lane_info == "LANE 1":
                if left_white_pixel > left_yellow_pixel:
                    lane_info = "LANE 2"
                    
                above_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi3, cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow))
                yellow_mask = cv2.inRange(roi3, lower_yellow, upper_yellow)
                contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if above_yellow_pixel >= crossing_centerline_thres:
                    lane_info = "Possible The Crossing CenterLine"
            
            elif lane_info == "Possible The Crossing CenterLine":
                if right_white_pixel > right_yellow_pixel:
                    lane_info = "LANE 3"
                
                elif right_white_pixel < right_yellow_pixel:
                    lane_info = "Encroaching The CenterLine"
            
            elif lane_info == "Encroaching The CenterLine":
                above_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi3, cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow))
                yellow_mask = cv2.inRange(roi3, lower_yellow, upper_yellow)
                contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) >= crossing_centerline_thres and above_yellow_pixel >= crossing_centerline_thres:
                    lane_info = "Possible The Crossing CenterLine"
            
            elif lane_info == "Crossing The Pedestrian Crosswalk":
                if len(crosswalk_exist) >= 2 and len(crosswalk_exist) < 4:
                    lane_info = temp_lane_info
                    start_time = time.time()
            
        elif len(crosswalk_exist) >= 10:
            if lane_info != "Crossing The Pedestrian Crosswalk":
                temp_lane_info = lane_info
            lane_info = "Crossing The Pedestrian Crosswalk"
            start_time = time.time()
            flag = True
    
    elif flag:
        lane_info = "Crossing The Pedestrian Crosswalk"
        if time.time() - start_time >= 1:
            flag = False
        
    # if len(crosswalk_exist) >= 0:
    #     cv2.putText(frame, "Rectangle Num :" + str(len(crosswalk_exist)), (position[0] + 100, position[1] + 100), font, font_scale, font_color, line_type)
    cv2.putText(frame, lane_info, position, font, font_scale, font_color, line_type)
    out.write(frame)
    cv2.imshow('Result', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("=" * 20)
        print("KeyBoard Interrupt")
        print("=" * 20)
        break

cap.release()
out.release()
cv2.destroyAllWindows()
