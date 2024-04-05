import cv2
import numpy as np

video_path = r"C:\Users\User\Desktop\MORARI 4월부터 차선검출 코드\LaneDetection\no_gps#5.mp4"
cap = cv2.VideoCapture(video_path)
if cap is None:
    exit()
    
# == Img processing == 
def process(frame):
    blur_frame = cv2.bilateralFilter(frame, d=3, sigmaColor=10, sigmaSpace=5)
    hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # White
    white_lower_thres = np.uint8([0, 0, 200])
    white_upper_thres = np.uint8([255, 255, 255])
    
    # Yellow
    yellow_lower_thres = np.uint8([20, 100, 100])
    yellow_upper_thres = np.uint8([30, 255, 255])
    
    white_mask = cv2.inRange(hsv_frame, white_lower_thres, white_upper_thres)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower_thres, yellow_upper_thres)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

# == Interpolation == 
def interpolation_lane(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.dilate(frame, kernel)
    
    return dst

# == ROI 설정 == 
def roi(frame, origin_frame):
    
    left_mask = np.zeros_like(frame)
    right_mask = np.zeros_like(frame)
    
    if len(frame.shape) > 2:
        channel_cnt = frame.shape[2]
        ignore_mask_color = (255, ) * channel_cnt
        
    else:
        ignore_mask_color = 255
    
    height, width = frame.shape[:2]
    
    
    # 왼쪽 차선
    left_lane_bottom_left = [0, height * 0.95]
    left_lane_top_left = [width * 0.4, height * 0.6]
    left_lane_bottom_right = [width * 0.4, height * 0.95]
    left_lane_top_right = [width * 0.5, height * 0.6]
    
    left_lane_vertices = np.array([[left_lane_bottom_left, left_lane_top_left, left_lane_top_right, left_lane_bottom_right]],
                                dtype=np.int32) 
    
    cv2.fillPoly(left_mask, left_lane_vertices, ignore_mask_color)
    left_masked_img = cv2.bitwise_and(frame, left_mask)
    
    # 오른쪽 차선
    right_lane_bottom_left = [width * 0.7, height * 0.95]
    right_lane_top_left = [width * 0.6, height * 0.6]
    right_lane_bottom_right = [width, height * 0.95]
    right_lane_top_right = [width * 0.7, height * 0.6]
    
    right_lane_vertices = np.array([[right_lane_bottom_left, right_lane_top_left, right_lane_top_right, right_lane_bottom_right]],
                                   dtype = np.int32)
    
    cv2.fillPoly(right_mask, right_lane_vertices, ignore_mask_color)
    right_masked_img = cv2.bitwise_and(frame, right_mask)
    
    result_masked_frame = cv2.bitwise_or(left_masked_img, right_masked_img)
    
    # ROI 설정 확인
    # 오른쪽
    cv2.line(origin_frame, right_lane_vertices[0][0], right_lane_vertices[0][1], (0, 255, 0), 2)
    cv2.line(origin_frame, right_lane_vertices[0][1], right_lane_vertices[0][2], (0, 255, 0), 2)
    cv2.line(origin_frame, right_lane_vertices[0][2], right_lane_vertices[0][3], (0, 255, 0), 2)
    cv2.line(origin_frame, right_lane_vertices[0][3], right_lane_vertices[0][0], (0, 255, 0), 2)
    
    cv2.line(origin_frame, left_lane_vertices[0][0], left_lane_vertices[0][1], (0, 255, 0), 2)
    cv2.line(origin_frame, left_lane_vertices[0][1], left_lane_vertices[0][2], (0, 255, 0), 2)
    cv2.line(origin_frame, left_lane_vertices[0][2], left_lane_vertices[0][3], (0, 255, 0), 2)
    cv2.line(origin_frame, left_lane_vertices[0][3], left_lane_vertices[0][0], (0, 255, 0), 2)
    
    return result_masked_frame, origin_frame


# 내가 지금 몇 차선으로 달리고 있나요??
def left_detect(frame):
    height, width, _ = frame.shape
    current_left = None
    
    #  == 왼쪽 차선 == 
    left_lane_bottom_left = [0, height * 0.95]
    left_lane_top_left = [int(width * 0.4), int(height * 0.6)]
    left_lane_bottom_right = [int(width * 0.4), int(height * 0.95)]
    left_lane_top_right = [int(width * 0.5), int(height * 0.6)]
    
    roi_pts = np.array([left_lane_bottom_left, left_lane_top_left, left_lane_top_right, left_lane_bottom_right], np.int32)
    
    # =======================================
    # 흰색?
    white_lower_thres = np.array([0, 0, 200], dtype=np.uint8)
    white_upper_thres = np.array([255, 255, 255], dtype=np.uint8)
    left_white_mask = np.zeros_like(frame)  
    cv2.fillPoly(left_white_mask, [roi_pts], (255, 255, 255))
    
    white_mask_range = cv2.inRange(frame, white_lower_thres, white_upper_thres)
    
    left_lane_white = cv2.bitwise_and(white_mask_range, left_white_mask[:,:,0])
    
    left_num_white_pixels = np.count_nonzero(left_lane_white)
    # =======================================
    
    # =======================================
    # 노란색?
    yellow_lower_thres = np.array([0, 100, 100], dtype=np.uint8)
    yellow_upper_thres = np.array([80, 255, 255], dtype=np.uint8)
    
    
    left_yellow_mask = np.zeros_like(frame)
    cv2.fillPoly(left_yellow_mask, [roi_pts], (255, 255, 255))
    
    yellow_mask_range = cv2.inRange(frame, yellow_lower_thres, yellow_upper_thres)
    left_lane_yellow = cv2.bitwise_and(yellow_mask_range, left_yellow_mask[:,:,0])
    
    left_num_yellow_pixels = cv2.countNonZero(left_lane_yellow)
    # =======================================

    # 좌 차선 무슨 색이게?
    if left_num_yellow_pixels > left_num_white_pixels:
        current_left_color = "Yellow"
    else:
        current_left_color = "White"
    
    print('White', left_num_white_pixels)
    print('Yellow', left_num_yellow_pixels)
    print("Left Lane", current_left_color)
    cv2.imshow('left_lane_white', frame)
    
    return current_left_color


while True:
    ret, frame = cap.read()
    
    if ret is None:
        print('No Frame')
        break
    processed_frame = process(frame)
    interpolation_frame = interpolation_lane(processed_frame)
    roi_result_masked_frame, roi_origin_frame = roi(interpolation_frame ,frame)
    
    current_left = left_detect(roi_result_masked_frame)
    
    print(current_left)
    
    cv2.imshow('Frame', roi_result_masked_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()