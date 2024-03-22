import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

class cone_detect:
    def __init__(self, PATH):
        self.img = cv2.imread(PATH)
        if self.img is None:
            print('Img Load Error !!!!!!')
            sys.exit()
        else:
            print(f'img shape : {self.img.shape}')
            
    def enhance_red_mask(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red mask
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        
        red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)    
        red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        # ======================================================
        
        # White mask
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([179, 30, 255])
        
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
        # ======================================================
        
        result_mask = cv2.bitwise_or(red_mask, white_mask)
        
        result= cv2.bitwise_and(frame, frame, mask = result_mask)
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.bilateralFilter(result, 9, 75, 100)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)
        
        return result
        
        
        
    
    def roi_img(self, frame):
        mask = np.zeros_like(frame)
        height, width = frame.shape[:2]
        
        bottom_left = [width * 0.05, height]
        top_left = [width * 0.1, height * 0.5]
        bottom_right = [width * 0.8, height]
        top_right = [width * 0.8, height * 0.5]
        
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype = np.int32)
        
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        masked_img = cv2.bitwise_and(frame, mask)
        
        
        return masked_img
    
    def run(self):
        red_img = self.enhance_red_mask(self.img)
        roi_img = self.roi_img(red_img)
        
        
        cv2.imshow('red_img', red_img)
        cv2.imshow('roi_img', roi_img)
        
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    
    
def main():
    PATH = 'cone_detection\cone1.jpg'
    cone_detect_system = cone_detect(PATH)
    cone_detect_system.run()
    
if __name__ == "__main__":
    main()