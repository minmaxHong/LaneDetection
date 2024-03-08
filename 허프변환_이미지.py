import numpy as np
import cv2
import sys

class ImageProcessing:
    def __init__(self, PATH,
                 bottom_left, top_left, top_right, bottom_right):
        self.img = cv2.imread(PATH)
        self.ordinal_img = cv2.imread(PATH)
        self.crop_pts = np.array([[bottom_left,
                                  top_left,
                                  top_right,
                                  bottom_right]], dtype = np.int32)
        
        # 예외 처리
        if self.img is None:
            sys.exit()
    
    #  == RGB -> HSV ==
    def rgb2hsv(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
    
    # == HSV -> GRAY ==
    def convert_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # == 이미지에서 노이즈 제거 == 
    def blur_img(self, img, kernel_size = 5):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    #  == HSV에서 임의의 노란색, 흰색만 선택하기 ==
    def hsv_color_selection(self):
        hsv_img = self.rgb2hsv()
        
        '''
        차선의 색은 hyperparameter값임
        '''
        # 흰색 차선 
        white_lower_thres = np.uint8([0, 0, 200])
        white_upper_thres = np.uint8([255, 30, 255])
        
        # 노란색 차선
        yellow_lower_thres = np.uint8([20, 100, 100])
        yellow_upper_thres = np.uint8([30, 255, 255])
        
        white_mask = cv2.inRange(hsv_img, white_lower_thres, white_upper_thres)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower_thres, yellow_upper_thres)
        
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        masked_img = cv2.bitwise_and(self.img, self.img, mask = mask)
        
        return masked_img
    
    # == 관심 영역 설정 == 
    def roi(self, img):
        mask = np.zeros_like(img)
        
        if len(img.shape) > 2:
            channel_cnt = img.shape[2]
            ignore_mask_color = (255, ) * channel_cnt
            
        else:
            ignore_mask_color = 255        
        
        cv2.fillPoly(mask, self.crop_pts, ignore_mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        
        return masked_img
    
    # == 버드아이뷰 == --> 곡률을 계산하기 위해서 사용한다.
    def bird_eye_view(self, img):
        height, width = img.shape[:2]
        
        # 좌/상, 좌/하, 우/상, 우/하
        corner_points_arr = np.float32(self.crop_pts)
        
        img_p1 = [0, 0] # 좌/상
        img_p2 = [0, height] # 우/하
        img_p3 = [width, height] # 우/상
        img_p4 = [width, 0] # 좌/하
        
        img_params = np.float32([img_p1, img_p2, img_p3, img_p4])
        
        mat = cv2.getPerspectiveTransform(corner_points_arr, img_params)
        img_transformed = cv2.warpPerspective(img, mat, (width, height))
        
        return img_transformed
    
    # == 허프 변환 == 
    def hough_transform_lines(self, img):
        rho = 1
        theta = np.pi / 180
        thres = 20
        minLineLength = 20
        maxLineGap = 300
        
        lines = cv2.HoughLinesP(img, rho = rho, theta = theta, threshold = thres,
                                minLineLength = minLineLength, maxLineGap = maxLineGap)
        
        return lines
    
    
    # == 허프변환을 통해서 이미지와 차선 검출을 받고, 선 그리기 == 
    def draw_hough_lines(self, img, lines):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color = (0, 255, 0), thickness = 2)
                
        return img
        
# # ==  좌표값을 알기 위한 callback 함수 == 
# def mouse_callback(event, x, y , flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"해당 좌표 : {x} {y}")
        
if __name__ == "__main__":
    PATH = "lane.jpg"
    
    bottom_left = [0, 535]
    top_left = [300, 380]
    top_right = [470, 380]
    bottom_right = [800, 535]
    IP = ImageProcessing(PATH,
                         bottom_left = bottom_left,
                         top_left = top_left,
                         top_right = top_right,
                         bottom_right = bottom_right)
    
    
    to_white_yellow_hsv_img = IP.hsv_color_selection()
    to_gray_img = IP.convert_gray(to_white_yellow_hsv_img)
    to_blur_img = IP.blur_img(to_gray_img)
    to_roi_img = IP.roi(to_blur_img)
    to_bird_eye_img = IP.bird_eye_view(to_roi_img)
    hough_lines = IP.hough_transform_lines(to_roi_img)
    to_draw_hough_lines = IP.draw_hough_lines(cv2.imread(PATH), hough_lines)
    
    # == 좌표 찾으려고 선 그리기 == 
    # ptss = [(0, 535), (300, 380), (470, 380), (800, 535)]
    
    # for pts in ptss:
    #     cv2.circle(to_blur_img, pts, 5, (0, 255, 0), -1)
    
    # cv2.line(to_blur_img, ptss[0], ptss[1], (255, 0, 0), 2)    
    # cv2.line(to_blur_img, ptss[1], ptss[2], (255, 0, 0), 2)    
    # cv2.line(to_blur_img, ptss[2], ptss[3], (255, 0, 0), 2)    
    # cv2.line(to_blur_img, ptss[3], ptss[0], (255, 0, 0), 2)    
    # cv2.setMouseCallback('img', mouse_callback)    

    result = cv2.hconcat([cv2.imread(PATH), to_draw_hough_lines])
    result = cv2.resize(result, (640, 480))
    cv2.imshow('img', result)
    cv2.waitKey()        
    cv2.destroyAllWindows()