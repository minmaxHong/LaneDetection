#!/usr/bin/env python3
#-*-coding:utf-8-*-

# Python packages

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError
import numpy as np
import cv2

class StopLaneDetector:
    def __init__(self, stopline_thres = 200):
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed2", CompressedImage, self.callback)
        self.stopline_thres = stopline_thres
        self.img = None
        self.count = 1
        self.detect = False
        
        
    def callback(self, msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)
            
        self.img = img_bgr
        
        
        if self.img is None:
            raise ValueError("Load Img Error")
        
    def roi(self, img):
        mask = np.zeros_like(img)
        ignore_mask_color = 255
        
        height, width = img.shape[:2]
        bottom_left = [width * 0.125, height * 0.95]
        top_left = [width * 0.125, height * 0.85]
        bottom_right = [width * 0.875, height * 0.95]
        top_right = [width * 0.875, height * 0.85]
        
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32) 
        
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        
        return masked_img 
    
    def process(self, img):
        blur_img = cv2.GaussianBlur(img, (5, 5), 0)
        _, L, _ = cv2.split(cv2.cvtColor(blur_img, cv2.COLOR_BGR2HLS))
        _, lane = cv2.threshold(L, self.stopline_thres, 255, cv2.THRESH_BINARY)
     
        return lane
    
    def white_histogram(self, img):
        histogram = np.sum(img[int(img.shape[0] / 2) :, :], axis=0)
        
        return histogram
    
    def uniform_histogram(self, histogram, threshold = 1200):
        uniformity_score = np.std(histogram)
        flag = uniformity_score < threshold
        
        return flag  
    
    def run(self):
        if self.img is not None:
            process_img = self.process(self.img)
            roi_img = self.roi(process_img)
            histogram = self.white_histogram(roi_img)
            detect_white = self.uniform_histogram(histogram)

            lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=180)

            if lines is not None:
                slopes = self.zero_slope(lines)
                avg_slope = np.mean(slopes)

                if abs(avg_slope) < 0.001 and np.sum(histogram) > 80000 and detect_white:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        self.detect = True
                        self.count += 1
                        print(f'Detect Stop Line!!{self.count}')
                        return True
                        
            else:      
                print('Not Detect!!')
                return False
            
            cv2.imshow('img', self.img)
            # cv2.imshow('process', process_img)
            # cv2.imshow('roi_img', roi_img)
        # cv2.destroyAllWindows()
            
    
    @staticmethod
    def zero_slope(lines):
        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            try:
                slope = (y2 - y1) / (x2 - x1)
            except:
                slope = (y2 - y1) / (x2 - x1 + 1e-9)
            slopes.append(slope)
            
        return slopes

def main():
    rospy.init_node('StopLaneDetector', anonymous=True)
    sd = StopLaneDetector()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        sd.run()
        rate.sleep()

if __name__ == "__main__":
    main()
    