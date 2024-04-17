import numpy as np
import cv2
# import taedx_order_crosswalk
from matplotlib import pyplot as plt


video_path = 'output_clip.mp4'
cap = cv2.VideoCapture(video_path)
if cap is None:
    print('==========================')
    print('NO Cap')
    print('==========================')
    exit()

# == bird eye view == 
# 1. 멀리 있는 정지선만 검출
def compute_perspective(frame):
    persp_frame = frame.copy()  # 입력 프레임 복사본 생성
    height, width, _ = persp_frame.shape

    left_bottom = [0, 350] 
    left_top = [0, 230]
    right_bottom = [width, 350]
    right_top = [width, 230]
    
    persp_trapezoid_coord = [(left_bottom[0], left_bottom[1]), (left_top[0], left_top[1]),
                            (right_top[0], right_top[1]), (right_bottom[0], right_bottom[1])]
    

    src = np.float32([left_bottom, left_top, right_bottom, right_top])
    dst = np.float32([[0, height - 1], [0, 0],
                        [width - 1, height - 1], [width - 1, 0]])
    
    persp_correction = cv2.getPerspectiveTransform(src, dst)
    frame2persp = cv2.warpPerspective(persp_frame, persp_correction, (width , height), flags = cv2.INTER_LANCZOS4)
    frame2persp_enhanceframe = cv2.normalize(frame2persp, None, 0, 255, cv2.NORM_MINMAX)
    frame2persp_enhanceframe_thresh = cv2.threshold(frame2persp_enhanceframe, 200, 255, cv2.THRESH_BINARY)[1]

    cv2.line(frame2persp_enhanceframe, persp_trapezoid_coord[0], persp_trapezoid_coord[1],  (0, 255, 0), 2)
    cv2.line(frame2persp_enhanceframe, persp_trapezoid_coord[1], persp_trapezoid_coord[2],  (0, 255, 0), 2)
    cv2.line(frame2persp_enhanceframe, persp_trapezoid_coord[2], persp_trapezoid_coord[3],  (0, 255, 0), 2)
    cv2.line(frame2persp_enhanceframe, persp_trapezoid_coord[3], persp_trapezoid_coord[0],  (0, 255, 0), 2)
    
      # 하얀색만 남기기
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(frame2persp_enhanceframe_thresh, lower_white, upper_white)
    result = cv2.bitwise_and(frame2persp_enhanceframe_thresh, frame2persp_enhanceframe_thresh, mask=mask)
    
    return frame2persp_enhanceframe

# == 히스토그램 계산 == 
def white_histogram(frame):
    histogram = np.sum(frame[int(frame.shape[0] / 2) :, :], axis=0)
    
    return histogram

# == 관심영역 == 
def roi(frame, thresh):
    close_mask = np.zeros_like(thresh)
    
    ignore_mask = 255
    
    height, width = frame.shape[:2]
    
    # 가까이 있는 것
    close_bottom_left = [0, height]
    close_top_left = [0, height * 0.7]
    close_bottom_right = [width, height]
    close_top_right = [width, height * 0.7]
    
    close_vertice = np.array([[close_bottom_left, close_top_left,
                               close_top_right, close_bottom_right]], dtype = np.int32)
    
    cv2.line(frame, close_vertice[0][0], close_vertice[0][1],  (0, 255, 0), 2)
    cv2.line(frame, close_vertice[0][1], close_vertice[0][2],  (0, 255, 0), 2)
    cv2.line(frame, close_vertice[0][2], close_vertice[0][3],  (0, 255, 0), 2)
    cv2.line(frame, close_vertice[0][3], close_vertice[0][0],  (0, 255, 0), 2)
    
    # 가까이 있는 것 fillPoly
    cv2.fillPoly(close_mask, close_vertice, ignore_mask)
    close_masked_img = cv2.bitwise_and(thresh, close_mask)
    
    roi_histogram = white_histogram(close_masked_img)
    
    return frame, roi_histogram


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    persp_frame = compute_perspective(frame)
    frame_with_roi, roi_hist = roi(frame, persp_frame)

    cv2.imshow('Persp', persp_frame)
    # cv2.imshow('Frame with ROI', frame_with_roi)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
