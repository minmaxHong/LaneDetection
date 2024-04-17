import cv2
import numpy as np

def persp_transform(frame):
    height, width = frame.shape[:2]
    
    # src : 관심영역, dst : 도착 지점
    # 왼쪽 상단, 오른쪽 상단, 왼쪽 하단, 오른쪽 상단
    src_bottom_left = [width * 0.1, height * 0.95]
    src_top_left = [width * 0.4, height * 0.6]
    src_bottom_right = [width * 0.9, height * 0.95]
    src_top_right = [width * 0.6, height * 0.6]
    
    dst_bottom_left = [0, height]
    dst_top_left = [0 ,0]
    dst_bottom_right = [width, 0]
    dst_top_right = [width, height]
    
    src_points = np.float32([src_bottom_left, src_top_left, src_top_right, src_bottom_right])
    dst_points = np.float32([dst_bottom_left, dst_top_left, dst_bottom_right, dst_top_right])
    
    mat = cv2.getPerspectiveTransform(src_points, dst_points)
    bed_frame = cv2.warpPerspective(frame, mat, (width, height))
    
    return bed_frame