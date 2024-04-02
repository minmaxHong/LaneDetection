import numpy as np
import cv2
import sys

 # height, width, channel : (600, 1024, 3)
img = cv2.imread('sd1.jpg')
if img is None:
    print('No Img')
    sys.exit()

height, width = img.shape[:2]

# == pixel 임계값 조정 ==
def threshold(edge_img):
    binary_img = np.zeros_like(edge_img)
    height = edge_img.shape[0]
    
    lower_thres = 15
    upper_thres = 60
    delta_thres = lower_thres - upper_thres
    
    # height가 작을수록 차선이 잘 안보이고, 클수록 차선이 잘 보임 -> 이를 지수가중평균으로 계산
    for y in range(height):
        edge_line = edge_img[y, :]
        line_thres = upper_thres + delta_thres * (y / height)
        
        binary_img[y, edge_line > line_thres] = 255
    
    return binary_img

# == BGR에서 HLS 바꾸기 ==
# 1. H, S는 noise가 많기 때문에 사용하기 어렵다.
def convert_hls(img):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    return hls_img

def rm_noise(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
        
    return img 


# == 원근변환 값 구하기 ==
def compute_perspective(img, width, height, left_bottom, left_top, right_top, right_bottom):
    persp_trapezoid_coord = [(left_bottom[0], left_bottom[1]), (left_top[0], left_top[1]),
                             (right_top[0], right_top[1]), (right_bottom[0], right_bottom[1])]
    src = np.float32([left_bottom, left_top, right_bottom, right_top])
    
    # width 
    small_width = left_bottom[0]
    larger_width = right_bottom[0]
    
    warp_width = larger_width - small_width
    
    # height
    small_height = left_top[1]
    larger_height = left_bottom[1]
    
    warp_height = larger_height - small_height
    
    dst = np.float32([[0, height - 1], [0, 0], # left_bottom, left_top
                     [width - 1, height - 1], [width - 1, 0]]) # right_bottom, right_top
    
    persp_correction = cv2.getPerspectiveTransform(src, dst)
    persp_correction_inv = cv2.getPerspectiveTransform(dst, src)
    
    persp_img = cv2.warpPerspective(img, persp_correction, (width, height), flags = cv2.INTER_LANCZOS4) # high frequency 지켜주는 interpolation
    # # src
    # cv2.line(img, persp_trapezoid_coord[0], persp_trapezoid_coord[1],  (0, 255, 0), 2)
    # cv2.line(img, persp_trapezoid_coord[1], persp_trapezoid_coord[2],  (0, 255, 0), 2)
    # cv2.line(img, persp_trapezoid_coord[2], persp_trapezoid_coord[3],  (0, 255, 0), 2)
    # cv2.line(img, persp_trapezoid_coord[3], persp_trapezoid_coord[0],  (0, 255, 0), 2)
    
    # # # dst
    # cv2.circle(img, (int(dst[0][0]), int(dst[0][1])), 2, (255, 255, 255), 3)
    # cv2.circle(img, (int(dst[1][0]), int(dst[1][1])), 2, (255, 255, 255), 3)
    # cv2.circle(img, (int(dst[2][0]), int(dst[2][1])), 2, (255, 255, 255), 3)
    # cv2.circle(img, (int(dst[3][0]), int(dst[3][1])), 2, (255, 255, 255), 3)
    
    return persp_img

# == 엣지 검출 & HLS 변환== 
def edge_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(img) # H, S는 noise가 너무 심해서 버림
    
    edge_x = cv2.Scharr(L, cv2.CV_64F, 1, 0) # 64-bit float 형태
    edge_x = np.absolute(edge_x)
    
    return np.uint8(255 * (edge_x / np.max(edge_x))) # 0 ~ 255사이로 변경



persp_img = compute_perspective(img, width, height, [160, 425], [484, 310], [546, 310], [877, 425])
rm_img = rm_noise(persp_img)
edge_img = edge_detection(persp_img)
thres_img = threshold(edge_img)

# cv2.imshow('Edge', edge_img)
cv2.imshow('Thres', thres_img)


cv2.waitKey()
cv2.destroyAllWindows()
