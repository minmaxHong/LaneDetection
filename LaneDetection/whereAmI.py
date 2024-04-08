import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\H_\Desktop\4월1일 모라이 차선검출\몇 차선\no_gps#13.avi"
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

x1, y1, width1, height1 = int(frame.shape[1] * 0.3), int(frame.shape[0] * 0.8), 200, 200
x2, y2, width2, height2 = int(frame.shape[1] * 0.725), int(frame.shape[0] * 0.8), 200, 200

lower_white = np.array([0, 0, 200], dtype=np.uint8)
upper_white = np.array([180, 30, 255], dtype=np.uint8)

lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

lane_info = "LANE 3" 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    roi1 = frame[y1:y1+height1, x1:x1+width1]
    roi2 = frame[y2:y2+height2, x2:x2+width2]

    left_white_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV), lower_white, upper_white))
    left_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow))

    right_white_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV), lower_white, upper_white))
    right_yellow_pixel = cv2.countNonZero(cv2.inRange(cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV), lower_yellow, upper_yellow))

    cv2.rectangle(frame, (x1, y1), (x1+width1, y1+height1), (0, 255, 0), 2)
    cv2.rectangle(frame, (x2, y2), (x2+width2, y2+height2), (0, 255, 0), 2)

    # print("첫 번째 ROI의 흰색 픽셀 개수:", left_white_pixel)
    # print("첫 번째 ROI의 노란색 픽셀 개수:", left_yellow_pixel)
    # print("두 번째 ROI의 흰색 픽셀 개수:", right_white_pixel)
    # print("두 번째 ROI의 노란색 픽셀 개수:", right_yellow_pixel)

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
    
    position = (640, 320)  
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  
    line_type = 2

    cv2.putText(frame, lane_info, position, font, font_scale, font_color, line_type)
    cv2.imshow('Result', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
